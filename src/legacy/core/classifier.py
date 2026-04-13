"""
image_safety_classifier.py
이미지 안전성 분류기
- 새로운 이미지에 대한 안전/위험 분류
- 다중 모델 앙상블 예측
- 신뢰도 점수 제공
- 실시간 예측 인터페이스
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import pickle
from datetime import datetime
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

from .embedders import get_embedder
from ..utils import analysis_utils as utils

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """예측 결과 클래스"""
    prediction: str  # 'safe' or 'danger'
    confidence: float  # 0.0 - 1.0
    probabilities: Dict[str, float]  # {'safe': 0.3, 'danger': 0.7}
    model_votes: Dict[str, str]  # 각 모델의 개별 예측
    model_confidences: Dict[str, float]  # 각 모델의 신뢰도
    danger_score: float  # 위험도 점수 (0.0 - 1.0)
    embedding_similarity: Dict[str, float]  # 참조 임베딩과의 유사도


class SafetyClassifier:
    """이미지 안전성 분류기"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.models = {}
        self.classifiers = {}
        self.danger_directions = {}
        self.reference_embeddings = {}
        self.is_trained = False
        self.training_stats = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """설정 로드"""
        default_config = {
            "device": "cuda:4",
            "model_types": ["siglip", "clip", "dino", "eva_clip"],
            "cache_base_path": "/workspace/prj_cctvPoc2_cv/tmp_ar_test",
            "data_base_path": "/workspace/prj_cctvPoc2_cv/tmp_ar_test/data/danger_al",
            "model_save_path": "./trained_models",
            "danger_threshold": 0.5,
            "confidence_threshold": 0.6,
            "ensemble_method": "voting",  # 'voting', 'averaging', 'stacking'
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _init_models(self):
        """임베딩 모델 초기화"""
        logger.info("임베딩 모델 초기화 중...")
        
        for model_type in self.config["model_types"]:
            try:
                model_config = {
                    "device": self.config["device"],
                    "model_type": model_type,
                    "cache_path": os.path.join(
                        self.config["cache_base_path"], 
                        f"embedding_cache_{model_type}.pkl"
                    ),
                }
                self.models[model_type] = get_embedder(model_config)
                logger.info(f"✅ {model_type} 모델 초기화 완료")
            except Exception as e:
                logger.error(f"❌ {model_type} 모델 초기화 실패: {e}")
    
    def train(self, train_data_path: str = None, validation_split: float = 0.2):
        """분류기 훈련"""
        logger.info("=== 안전성 분류기 훈련 시작 ===")
        
        if not self.models:
            self._init_models()
        
        # 데이터 로드
        data_path = train_data_path or self.config["data_base_path"]
        
        # 데이터 구조 확인 - run_complete_analysis.py와 동일한 로직
        if os.path.exists(os.path.join(data_path, "danger")) and os.path.exists(os.path.join(data_path, "safe")):
            # 직접 danger, safe 폴더가 있는 경우
            danger_dir = os.path.join(data_path, "danger")
            safe_dir = os.path.join(data_path, "safe")
        elif os.path.exists(os.path.join(data_path, "danger_al")):
            # danger_al 폴더 안에 있는 경우
            danger_dir = os.path.join(data_path, "danger_al", "danger")
            safe_dir = os.path.join(data_path, "danger_al", "safe")
        else:
            # 기본값
            danger_dir = os.path.join(data_path, "danger")
            safe_dir = os.path.join(data_path, "safe")
        
        if not os.path.exists(danger_dir) or not os.path.exists(safe_dir):
            raise FileNotFoundError(f"훈련 데이터 디렉터리를 찾을 수 없습니다: {data_path}")
        
        # 데이터 수집
        danger_dict = utils.collect_by_key(danger_dir)
        safe_dict = utils.collect_by_key(safe_dir)
        common_keys = list(set(danger_dict.keys()) & set(safe_dict.keys()))
        
        logger.info(f"훈련 데이터: {len(common_keys)}개 키")
        
        # 각 모델별로 훈련
        for model_type, model in self.models.items():
            logger.info(f"--- {model_type} 모델 훈련 중 ---")
            
            try:
                # 특징 추출
                X, y, keys = self._extract_features(
                    model, danger_dict, safe_dict, common_keys
                )
                
                if len(X) == 0:
                    logger.warning(f"{model_type}: 유효한 훈련 데이터가 없습니다")
                    continue
                
                # 위험 방향 계산
                self.danger_directions[model_type] = self._compute_danger_direction(
                    model, danger_dict, safe_dict, common_keys
                )
                
                # 참조 임베딩 저장
                self.reference_embeddings[model_type] = {
                    'danger_center': self._compute_center_embedding(model, danger_dict),
                    'safe_center': self._compute_center_embedding(model, safe_dict)
                }
                
                # 분류기 훈련
                classifier = self._train_classifier(X, y, model_type)
                self.classifiers[model_type] = classifier
                
                # 성능 평가
                self._evaluate_classifier(classifier, X, y, model_type)
                
                logger.info(f"✅ {model_type} 훈련 완료")
                
            except Exception as e:
                logger.error(f"❌ {model_type} 훈련 실패: {e}")
        
        self.is_trained = True
        
        # 모델 저장
        self.save_models()
        
        logger.info("✅ 분류기 훈련 완료")
    
    def _extract_features(self, model, danger_dict, safe_dict, common_keys):
        """특징 추출"""
        X, y, keys = [], [], []
        
        # Danger 샘플
        for key in common_keys:
            try:
                danger_emb = model.extract_embeddings(danger_dict[key])
                if len(danger_emb) > 0:
                    avg_emb = np.mean(danger_emb, axis=0)
                    if not np.isnan(avg_emb).any():
                        X.append(avg_emb)
                        y.append(1)  # danger = 1
                        keys.append(key)
            except Exception as e:
                logger.debug(f"Danger 키 {key} 처리 중 오류: {e}")
        
        # Safe 샘플
        for key in common_keys:
            try:
                safe_emb = model.extract_embeddings(safe_dict[key])
                if len(safe_emb) > 0:
                    avg_emb = np.mean(safe_emb, axis=0)
                    if not np.isnan(avg_emb).any():
                        X.append(avg_emb)
                        y.append(0)  # safe = 0
                        keys.append(key)
            except Exception as e:
                logger.debug(f"Safe 키 {key} 처리 중 오류: {e}")
        
        return np.array(X), np.array(y), keys
    
    def _compute_danger_direction(self, model, danger_dict, safe_dict, common_keys):
        """위험 방향 벡터 계산"""
        danger_diffs = []
        
        for key in common_keys:
            try:
                danger_avg, safe_avg = utils.compute_avg_embeddings(
                    (danger_dict[key], safe_dict[key]), model
                )
                
                if danger_avg is not None and safe_avg is not None:
                    diff = danger_avg - safe_avg
                    if not np.isnan(diff).any():
                        danger_diffs.append(diff)
            except Exception:
                continue
        
        if danger_diffs:
            return np.mean(danger_diffs, axis=0)
        else:
            return None
    
    def _compute_center_embedding(self, model, data_dict):
        """중심 임베딩 계산"""
        all_embeddings = []
        
        for key, paths in data_dict.items():
            try:
                embeddings = model.extract_embeddings(paths)
                if len(embeddings) > 0:
                    all_embeddings.extend(embeddings)
            except Exception:
                continue
        
        if all_embeddings:
            return np.mean(all_embeddings, axis=0)
        else:
            return None
    
    def _train_classifier(self, X, y, model_type):
        """분류기 훈련"""
        # 앙상블 분류기 생성
        classifiers = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(probability=True, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=1000))
        ]
        
        if self.config["ensemble_method"] == "voting":
            classifier = VotingClassifier(classifiers, voting='soft')
        else:
            # 기본으로 Random Forest 사용
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
        classifier.fit(X, y)
        return classifier
    
    def _evaluate_classifier(self, classifier, X, y, model_type):
        """분류기 성능 평가"""
        try:
            # 교차 검증
            cv_scores = cross_val_score(
                classifier, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='accuracy'
            )
            
            # 예측 및 성능 계산
            y_pred = classifier.predict(X)
            y_prob = classifier.predict_proba(X)[:, 1]
            
            # 성능 통계
            self.training_stats[model_type] = {
                'accuracy': float(np.mean(cv_scores)),
                'accuracy_std': float(np.std(cv_scores)),
                'auc_score': float(roc_auc_score(y, y_prob)),
                'classification_report': classification_report(y, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
                'n_samples': len(X),
                'n_features': X.shape[1]
            }
            
            logger.info(f"{model_type} 성능 - 정확도: {np.mean(cv_scores):.3f}±{np.std(cv_scores):.3f}, AUC: {roc_auc_score(y, y_prob):.3f}")
            
        except Exception as e:
            logger.error(f"{model_type} 성능 평가 중 오류: {e}")
    
    def predict(self, image_path: str) -> PredictionResult:
        """단일 이미지 예측"""
        if not self.is_trained:
            raise RuntimeError("모델이 훈련되지 않았습니다. train() 메서드를 먼저 호출하세요.")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        
        logger.info(f"이미지 예측 중: {image_path}")
        
        model_votes = {}
        model_confidences = {}
        danger_scores = []
        similarities = {}
        
        # 각 모델별 예측
        for model_type, model in self.models.items():
            if model_type not in self.classifiers:
                continue
                
            try:
                # 임베딩 추출
                embedding = model.extract_embeddings([image_path])
                if len(embedding) == 0:
                    logger.warning(f"{model_type}: 임베딩 추출 실패")
                    continue
                
                avg_embedding = np.mean(embedding, axis=0)
                
                # 분류기 예측
                classifier = self.classifiers[model_type]
                pred_proba = classifier.predict_proba([avg_embedding])[0]
                pred_class = classifier.predict([avg_embedding])[0]
                
                # 결과 저장
                model_votes[model_type] = 'danger' if pred_class == 1 else 'safe'
                model_confidences[model_type] = float(max(pred_proba))
                
                # 위험도 점수 계산
                if self.danger_directions.get(model_type) is not None:
                    danger_score = self._compute_danger_score(
                        avg_embedding, model_type
                    )
                    danger_scores.append(danger_score)
                
                # 참조 임베딩과의 유사도
                if model_type in self.reference_embeddings:
                    ref_embs = self.reference_embeddings[model_type]
                    if ref_embs['danger_center'] is not None and ref_embs['safe_center'] is not None:
                        danger_sim = cosine_similarity(
                            [avg_embedding], [ref_embs['danger_center']]
                        )[0][0]
                        safe_sim = cosine_similarity(
                            [avg_embedding], [ref_embs['safe_center']]
                        )[0][0]
                        similarities[f"{model_type}_danger"] = float(danger_sim)
                        similarities[f"{model_type}_safe"] = float(safe_sim)
                
            except Exception as e:
                logger.error(f"{model_type} 예측 중 오류: {e}")
        
        # 앙상블 예측
        final_prediction, final_confidence, probabilities = self._ensemble_predict(
            model_votes, model_confidences
        )
        
        # 위험도 점수 계산
        avg_danger_score = np.mean(danger_scores) if danger_scores else 0.5
        
        return PredictionResult(
            prediction=final_prediction,
            confidence=final_confidence,
            probabilities=probabilities,
            model_votes=model_votes,
            model_confidences=model_confidences,
            danger_score=avg_danger_score,
            embedding_similarity=similarities
        )
    
    def predict_batch(self, image_paths: List[str]) -> List[PredictionResult]:
        """배치 이미지 예측"""
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"배치 예측 중 오류 ({image_path}): {e}")
                # 기본 결과 반환
                results.append(PredictionResult(
                    prediction='unknown',
                    confidence=0.0,
                    probabilities={'safe': 0.5, 'danger': 0.5},
                    model_votes={},
                    model_confidences={},
                    danger_score=0.5,
                    embedding_similarity={}
                ))
        
        return results
    
    def _compute_danger_score(self, embedding: np.ndarray, model_type: str) -> float:
        """위험도 점수 계산"""
        danger_direction = self.danger_directions.get(model_type)
        if danger_direction is None:
            return 0.5
        
        # 위험 방향으로의 투영
        projection = np.dot(embedding, danger_direction)
        
        # 정규화 (0-1 범위)
        # 이는 훈련 데이터의 분포를 기반으로 조정될 수 있습니다
        score = 1 / (1 + np.exp(-projection))  # 시그모이드 함수
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _ensemble_predict(self, model_votes, model_confidences):
        """앙상블 예측"""
        if not model_votes:
            return 'unknown', 0.0, {'safe': 0.5, 'danger': 0.5}
        
        # 투표 기반 예측
        danger_votes = sum(1 for vote in model_votes.values() if vote == 'danger')
        safe_votes = sum(1 for vote in model_votes.values() if vote == 'safe')
        
        if danger_votes > safe_votes:
            final_prediction = 'danger'
        elif safe_votes > danger_votes:
            final_prediction = 'safe'
        else:
            # 동점일 경우 신뢰도 기반 결정
            avg_confidence_danger = np.mean([
                conf for model, conf in model_confidences.items() 
                if model_votes[model] == 'danger'
            ]) if danger_votes > 0 else 0
            
            avg_confidence_safe = np.mean([
                conf for model, conf in model_confidences.items() 
                if model_votes[model] == 'safe'
            ]) if safe_votes > 0 else 0
            
            final_prediction = 'danger' if avg_confidence_danger > avg_confidence_safe else 'safe'
        
        # 신뢰도 계산
        final_confidence = np.mean(list(model_confidences.values()))
        
        # 확률 계산
        total_votes = len(model_votes)
        probabilities = {
            'safe': safe_votes / total_votes,
            'danger': danger_votes / total_votes
        }
        
        return final_prediction, float(final_confidence), probabilities
    
    def save_models(self, save_path: str = None):
        """훈련된 모델 저장"""
        save_path = save_path or self.config["model_save_path"]
        os.makedirs(save_path, exist_ok=True)
        
        # 분류기 저장
        classifier_path = os.path.join(save_path, "classifiers.pkl")
        with open(classifier_path, 'wb') as f:
            pickle.dump(self.classifiers, f)
        
        # 위험 방향 저장
        danger_dir_path = os.path.join(save_path, "danger_directions.pkl")
        with open(danger_dir_path, 'wb') as f:
            pickle.dump(self.danger_directions, f)
        
        # 참조 임베딩 저장
        ref_emb_path = os.path.join(save_path, "reference_embeddings.pkl")
        with open(ref_emb_path, 'wb') as f:
            pickle.dump(self.reference_embeddings, f)
        
        # 훈련 통계 저장
        stats_path = os.path.join(save_path, "training_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=4)
        
        # 설정 저장
        config_path = os.path.join(save_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        
        logger.info(f"✅ 모델 저장 완료: {save_path}")
    
    def load_models(self, load_path: str = None):
        """저장된 모델 로드"""
        load_path = load_path or self.config["model_save_path"]
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"모델 디렉터리를 찾을 수 없습니다: {load_path}")
        
        # 분류기 로드
        classifier_path = os.path.join(load_path, "classifiers.pkl")
        if os.path.exists(classifier_path):
            with open(classifier_path, 'rb') as f:
                self.classifiers = pickle.load(f)
        
        # 위험 방향 로드
        danger_dir_path = os.path.join(load_path, "danger_directions.pkl")
        if os.path.exists(danger_dir_path):
            with open(danger_dir_path, 'rb') as f:
                self.danger_directions = pickle.load(f)
        
        # 참조 임베딩 로드
        ref_emb_path = os.path.join(load_path, "reference_embeddings.pkl")
        if os.path.exists(ref_emb_path):
            with open(ref_emb_path, 'rb') as f:
                self.reference_embeddings = pickle.load(f)
        
        # 훈련 통계 로드
        stats_path = os.path.join(load_path, "training_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.training_stats = json.load(f)
        
        # 임베딩 모델 초기화
        self._init_models()
        
        self.is_trained = True
        logger.info(f"✅ 모델 로드 완료: {load_path}")
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            "is_trained": self.is_trained,
            "model_types": list(self.models.keys()),
            "classifiers": list(self.classifiers.keys()),
            "training_stats": self.training_stats,
            "config": self.config
        }


def create_prediction_report(results: List[PredictionResult], image_paths: List[str]) -> Dict:
    """예측 결과 리포트 생성"""
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": len(results),
        "predictions": {},
        "summary": {
            "safe_count": 0,
            "danger_count": 0,
            "unknown_count": 0,
            "high_confidence_count": 0,
            "low_confidence_count": 0
        }
    }
    
    for i, (result, image_path) in enumerate(zip(results, image_paths)):
        report["predictions"][image_path] = {
            "prediction": result.prediction,
            "confidence": result.confidence,
            "probabilities": result.probabilities,
            "danger_score": result.danger_score,
            "model_votes": result.model_votes,
            "model_confidences": result.model_confidences
        }
        
        # 요약 통계 업데이트
        if result.prediction == 'safe':
            report["summary"]["safe_count"] += 1
        elif result.prediction == 'danger':
            report["summary"]["danger_count"] += 1
        else:
            report["summary"]["unknown_count"] += 1
        
        if result.confidence > 0.7:
            report["summary"]["high_confidence_count"] += 1
        elif result.confidence < 0.5:
            report["summary"]["low_confidence_count"] += 1
    
    return report