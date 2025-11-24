# placeRecommender.py

from typing import Dict
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from sqlalchemy.orm import Session

from models.db_models import (
    Place,
    User,
    UserPlaceLikes
)

class PlaceCFRecommender:
    def __init__(self):
        self.model = None
        self.dataset = None
        self.user_map = {}
        self.item_map = {}
       

    def build_model(self, db: Session):
        users = db.query(User).all()
        places = db.query(Place).all()
        interactions = db.query(UserPlaceLikes).all()

        if not users or not places:
            print("Place CF: 데이터 부족")
            return

        self.dataset = Dataset()

        user_ids = [u.id for u in users]
        place_ids = [p.id for p in places]

        self.dataset.fit(users=user_ids, items=place_ids)

        interaction_data = [(i.user_id, i.place_id) for i in interactions]
        interaction_matrix, _ = self.dataset.build_interactions(interaction_data)

        self.model = LightFM(loss="warp")
        self.model.fit(interaction_matrix, epochs=20, num_threads=4)

        # 매핑 저장
        self.user_map = self.dataset.mapping()[0]
        self.item_map = self.dataset.mapping()[2]

    def predict_for_user(self, user_id: int, top_k=300):
        if not self.model or user_id not in self.user_map:
            return []

        internal_user = self.user_map[user_id]

        place_ids = list(self.item_map.keys())
        internal_items = [self.item_map[i] for i in place_ids]

        scores = self.model.predict(internal_user, internal_items)

        ranked = list(zip(place_ids, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked[:top_k]
    
    
    def get_cf_candidates(self, user_id: int, top_k: int = 100, exclude_place_ids: set = None) -> Dict[int, float]:
        """
        특정 user에 대해 place CF 점수 상위 top_k 반환
        return: {place_id: cf_score}
        """
        if not self.model or not self.dataset:
            return {}

        if user_id not in self.user_map:
           return {}

        internal_user_id = self.user_map[user_id]
        place_ids = list(self.item_map.keys())
        internal_place_ids = [self.item_map[pid] for pid in place_ids]
        scores = self.model.predict(
            user_ids=internal_user_id,
            item_ids=internal_place_ids,
        )

        exclude_set = exclude_place_ids or set()
    
        scored = []
        for i, pid in enumerate(place_ids):
            if pid in exclude_set:
                continue
            scored.append((pid, float(scores[i])))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_scored = scored[:top_k]

        return {pid: score for pid, score in top_scored}
    
def batch_distance(user_coords, place_coords):
    user_lat, user_lon = np.radians(user_coords)

    place_lats = np.radians(place_coords[:, 0])
    place_lons = np.radians(place_coords[:, 1])

    dlat = place_lats - user_lat
    dlon = place_lons - user_lon

    a = np.sin(dlat/2)**2 + np.cos(user_lat) * np.cos(place_lats) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    earth_radius_km = 6371
    distances = earth_radius_km * c

    return distances