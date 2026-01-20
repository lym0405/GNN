"""
Product Matcher: 텍스트 유사도 기반 다중 상품 매칭
====================================================
기업의 '주요상품목록' 텍스트를 분석하여 IO 상품 코드에 매칭

핵심 아이디어:
- 주요상품목록: "철강재, 건축자재, 철근" → IO 코드 [06, 09, 11]
- 다중 매핑: 한 기업이 여러 상품을 생산하는 경우
- 텍스트 유사도: TF-IDF + Cosine Similarity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductMatcher:
    """
    주요상품목록 텍스트를 IO 상품 코드에 매칭
    
    Parameters
    ----------
    io_product_names : Dict[str, str]
        IO 상품 코드 → 상품명 매핑
        예: {'01': '농림수산품', '06': '1차 철강 제품', ...}
    """
    
    def __init__(self, io_product_names: Dict[str, str]):
        self.io_product_names = io_product_names
        self.io_codes = list(io_product_names.keys())
        self.io_names = list(io_product_names.values())
        
        # TF-IDF Vectorizer 초기화
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),  # 단어 및 2-gram
            min_df=1
        )
        
        # IO 상품명을 벡터화
        self.io_vectors = self.vectorizer.fit_transform(self.io_names)
        
        logger.info(f"✅ ProductMatcher 초기화")
        logger.info(f"   - IO 상품 개수: {len(self.io_codes)}")
    
    def match_products(
        self,
        product_text: str,
        top_k: int = 3,
        threshold: float = 0.1
    ) -> List[Tuple[str, float]]:
        """
        주요상품목록 텍스트를 IO 코드에 매칭
        
        Parameters
        ----------
        product_text : str
            주요상품목록 (예: "철강재, 건축자재, 철근")
        top_k : int
            상위 K개 매칭 결과 반환
        threshold : float
            최소 유사도 임계값 (0.1 이상만 반환)
        
        Returns
        -------
        matches : List[Tuple[str, float]]
            [(IO코드, 유사도), ...] 형태로 반환
            예: [('06', 0.85), ('09', 0.42), ...]
        """
        if not product_text or pd.isna(product_text):
            return []
        
        # 텍스트 전처리
        text = self._preprocess(product_text)
        
        if not text:
            return []
        
        # 벡터화
        try:
            text_vector = self.vectorizer.transform([text])
        except:
            return []
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(text_vector, self.io_vectors)[0]
        
        # 상위 K개 추출
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        matches = []
        for idx in top_indices:
            score = similarities[idx]
            if score >= threshold:
                code = self.io_codes[idx]
                matches.append((code, float(score)))
        
        return matches
    
    def _preprocess(self, text: str) -> str:
        """텍스트 전처리"""
        # 특수문자 제거 (쉼표, 괄호 등)
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        # 공백 정리
        text = ' '.join(text.split())
        
        return text
    
    def batch_match(
        self,
        df_firms: pd.DataFrame,
        col_product_text: str = '주요상품목록',
        col_multi_code: str = 'IO상품_다중_대분류_코드',
        use_multi_code: bool = True,
        top_k: int = 3
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        전체 기업 데이터프레임에 대해 일괄 매칭
        
        Parameters
        ----------
        df_firms : pd.DataFrame
            기업 정보 데이터프레임
        col_product_text : str
            주요상품목록 컬럼명
        col_multi_code : str
            다중 분류 코드 컬럼명 (존재 시 우선 사용)
        use_multi_code : bool
            다중 분류 코드를 우선 사용할지 여부
        top_k : int
            텍스트 매칭 시 상위 K개
        
        Returns
        -------
        firm_products : Dict[str, List[Tuple[str, float]]]
            사업자번호 → [(IO코드, 가중치), ...] 매핑
        """
        logger.info(f"🔍 기업별 상품 매칭 시작...")
        logger.info(f"   - 기업 수: {len(df_firms)}")
        
        firm_products = {}
        
        # 사업자번호 컬럼 찾기
        col_biz = next((c for c in df_firms.columns if '사업자등록번호' in c), df_firms.columns[0])
        
        for idx, row in df_firms.iterrows():
            biz_id = str(row[col_biz]).strip()
            
            # 1) 다중 분류 코드가 있으면 우선 사용
            if use_multi_code and col_multi_code in df_firms.columns:
                multi_code = row[col_multi_code]
                
                if pd.notna(multi_code) and str(multi_code).strip():
                    # 쉼표로 구분된 코드들 파싱
                    codes = [c.strip() for c in str(multi_code).split(',')]
                    # 동일 가중치 부여
                    weight = 1.0 / len(codes)
                    firm_products[biz_id] = [(code, weight) for code in codes if code]
                    continue
            
            # 2) 주요상품목록 텍스트 매칭
            if col_product_text in df_firms.columns:
                product_text = row[col_product_text]
                matches = self.match_products(product_text, top_k=top_k)
                
                if matches:
                    # 유사도 정규화 (합이 1이 되도록)
                    total_score = sum(score for _, score in matches)
                    normalized = [(code, score/total_score) for code, score in matches]
                    firm_products[biz_id] = normalized
                    continue
            
            # 3) 매칭 실패 시 빈 리스트
            firm_products[biz_id] = []
        
        logger.info(f"✅ 매칭 완료")
        logger.info(f"   - 매칭 성공: {sum(1 for v in firm_products.values() if v)}")
        logger.info(f"   - 매칭 실패: {sum(1 for v in firm_products.values() if not v)}")
        
        return firm_products


def create_io_product_dict(io_table_path: str = None) -> Dict[str, str]:
    """
    IO 테이블에서 상품 코드 → 상품명 딕셔너리 생성
    
    Parameters
    ----------
    io_table_path : str
        IO 테이블 CSV 경로 (A_33.csv)
    
    Returns
    -------
    io_dict : Dict[str, str]
        {'01': '농림수산품', '02': '광산품', ...}
    """
    if io_table_path:
        df_io = pd.read_csv(io_table_path, index_col=0)
        io_names = df_io.index.tolist()
    else:
        # 기본 33개 IO 산업 (대분류)
        io_names = [
            "농림수산품", "광산품", "음식료품", "섬유 및 가죽제품",
            "목재 및 종이, 인쇄", "석탄 및 석유제품", "화학제품",
            "비금속광물제품", "1차 금속제품", "금속제품",
            "컴퓨터, 전자 및 광학기기", "전기장비", "기계 및 장비",
            "운송장비", "기타 제조업 제품", "전력, 가스, 증기 및 공기조절",
            "수도, 하수 및 폐기물처리", "건설", "도소매 및 상품중개서비스",
            "운송서비스", "음식점 및 숙박서비스", "출판, 영상, 방송통신 및 정보서비스",
            "금융 및 보험서비스", "부동산서비스", "전문, 과학 및 기술서비스",
            "사업지원서비스", "공공행정, 국방 및 사회보장", "교육서비스",
            "보건 및 사회복지서비스", "예술, 스포츠 및 여가 관련 서비스",
            "협회 및 단체, 수리 및 기타 개인서비스", "가구 내 고용활동",
            "달리 분류되지 않는 자가소비생산활동"
        ]
    
    # 코드 생성 (01, 02, ..., 33)
    io_dict = {f"{i+1:02d}": name for i, name in enumerate(io_names)}
    
    return io_dict


if __name__ == "__main__":
    # 테스트
    io_dict = create_io_product_dict()
    matcher = ProductMatcher(io_dict)
    
    # 테스트 케이스
    test_cases = [
        "철강재, 건축자재, 철근",
        "전자제품, 반도체, 컴퓨터부품",
        "의류, 섬유제품, 가방",
        "음식료품, 가공식품, 음료"
    ]
    
    for text in test_cases:
        matches = matcher.match_products(text, top_k=3)
        print(f"\n텍스트: {text}")
        print(f"매칭 결과:")
        for code, score in matches:
            print(f"  - {code}: {io_dict[code]} (유사도: {score:.3f})")
