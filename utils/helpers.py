# utils/helpers.py

import time
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from typing import Tuple, Optional
from functools import lru_cache

# 지오코더 초기화 (전역으로 한 번만)
geolocator = Nominatim(user_agent="flicker_recommender")


@lru_cache(maxsize=500)  # 최대 500개 주소 캐싱
def geocode(address: str) -> Optional[Tuple[float, float]]:
    """
    주소를 위도/경도로 변환 (캐싱 적용)
    
    Args:
        address: 한국 주소 (예: "서울특별시 강남구", "인천광역시 남동구")
    
    Returns:
        (latitude, longitude) or None
    
    Note:
        - Nominatim 사용 (1초당 1회 제한)
        - 광역시/구 수준까지 정확
        - 동/번지 수준은 지원하지 않음
    """
    if not address:
        return None
    
    try:
        
        location = geolocator.geocode(address, timeout=10, language='ko')
        
        if location:
            print(f"✓ Geocoded: {address} → ({location.latitude:.6f}, {location.longitude:.6f})")
            return (location.latitude, location.longitude)
        else:
            print(f"✗ Not found: {address}")
            return None
    
    except Exception as e:
        print(f"✗ Geocoding failed for {address}: {e}")
        return None


def calculate_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    두 좌표 간 거리 계산 (km)
    Args:
        coord1: (위도, 경도)
        coord2: (위도, 경도)
    Returns:
        거리 (킬로미터)
    """
    try:
        return geodesic(coord1, coord2).kilometers
    except Exception as e:
        print(f"Distance calculation failed: {e}")
        return float('inf')


def normalize_score(score: float, min_score: float, max_score: float) -> float:
    """
    점수를 0~1 사이로 정규화
    
    Args:
        score: 정규화할 점수
        min_score: 최소값
        max_score: 최대값
    
    Returns:
        0.0 ~ 1.0 사이의 정규화된 점수
    """
    if max_score == min_score:
        return 0.5
    return (score - min_score) / (max_score - min_score)


def distance_to_score(distance_km: float, max_distance: float = 100.0) -> float:
    """
    거리를 점수로 변환 (가까울수록 높은 점수)
    
    Args:
        distance_km: 거리 (킬로미터)
        max_distance: 이 거리 이상은 점수 0 (기본 100km)
    
    Returns:
        0.0 ~ 1.0 사이의 점수
    
    Example:
        >>> distance_to_score(0)     # 0km
        1.0
        >>> distance_to_score(50)    # 50km
        0.5
        >>> distance_to_score(100)   # 100km
        0.0
        >>> distance_to_score(150)   # 150km
        0.0
    """
    if distance_km >= max_distance:
        return 0.0
    return 1.0 - (distance_km / max_distance)


def clear_geocode_cache():
    """
    지오코딩 캐시 초기화
    주소 데이터가 변경되었을 때 호출
    """
    geocode.cache_clear()
    print("Geocoding cache cleared")


def get_cache_info():
    """
    캐시 정보 조회
    
    Returns:
        캐시 통계 정보
    """
    info = geocode.cache_info()
    return {
        'hits': info.hits,
        'misses': info.misses,
        'size': info.currsize,
        'max_size': info.maxsize,
        'hit_rate': f"{info.hits / (info.hits + info.misses) * 100:.1f}%" if (info.hits + info.misses) > 0 else "0%"
    }