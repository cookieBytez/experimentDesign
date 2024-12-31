import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict
from datetime import datetime, timedelta

class SKNN_E:
    def __init__(self, n_neighbors=5, recent_sessions_window=30):
        """
        Initialize SKNN_E model
        
        Parameters:
        -----------
        n_neighbors : int
            Number of neighbors to use for KNN
        recent_sessions_window : int
            Number of days to consider for recent sessions
        """
        self.n_neighbors = n_neighbors
        self.recent_sessions_window = recent_sessions_window
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
        self.mlb = MultiLabelBinarizer()
        
    def split_train_validation(self, purchase_events, sessions):
        """
        Split data into training and validation sets based on 'valid' column
        
        Parameters:
        -----------
        purchase_events : pd.DataFrame
            DataFrame containing purchase events with 'valid' column
        sessions : pd.DataFrame
            DataFrame containing session data with 'valid' column
            
        Returns:
        --------
        tuple
            (train_purchases, val_purchases, train_sessions, val_sessions)
        """
        # Split purchase events
        train_purchases = purchase_events[purchase_events['valid'] == 0].drop('valid', axis=1)
        val_purchases = purchase_events[purchase_events['valid'] == 1].drop('valid', axis=1)
        
        # Split sessions
        train_sessions = sessions[sessions['valid'] == 0].drop('valid', axis=1)
        val_sessions = sessions[sessions['valid'] == 1].drop('valid', axis=1)
        
        return train_purchases, val_purchases, train_sessions, val_sessions
        
    def preprocess_data(self, purchase_events, sessions):
        """
        Preprocess the purchase and session data
        
        Parameters:
        -----------
        purchase_events : pd.DataFrame
            DataFrame containing purchase events
        sessions : pd.DataFrame
            DataFrame containing session data
        """
        purchase_events['event_time'] = pd.to_datetime(purchase_events['event_time'])
        sessions['action_time'] = pd.to_datetime(sessions['action_time'])

        self.latest_date = max(purchase_events['event_time'].max(), 
                             sessions['action_time'].max())
        
        # Filter recent sessions
        #cutoff_date = self.latest_date - timedelta(days=self.recent_sessions_window)
        #recent_sessions = sessions[sessions['action_time'] >= cutoff_date]
        
        return purchase_events, sessions
    
    def create_user_action_vectors(self, sessions):
        """
        Create user action vectors using max pooling over all actions
        
        Parameters:
        -----------
        sessions : pd.DataFrame
            DataFrame containing session data
        """
        
        def create_action_feature(row):
            features = []
            features.append(f"{row['action_section']}")
            features.append(f"{row['action_section']}_{row['action_type']}")
            if pd.notna(row['action_object']):
                features.append(f"{row['action_object']}")
                features.append(f"{row['action_section']}_{row['action_object']}")
                features.append(f"{row['action_section']}_{row['action_type']}_{row['action_object']}")
            return features
        
        # Group actions by user with multiple features per action
        user_actions = defaultdict(set)
        for _, row in sessions.iterrows():
            features = create_action_feature(row)
            user_actions[row['event_id']].update(features)
        
        # Convert to list for MultiLabelBinarizer
        user_actions_list = [(k, list(v)) for k, v in user_actions.items()]
        self.event_ids = [x[0] for x in user_actions_list]
        actions_list = [x[1] for x in user_actions_list]
        

        self.action_matrix = self.mlb.fit_transform(actions_list)
        self.feature_names = self.mlb.classes_
        
        return self.action_matrix
        
    
    
    def create_purchase_vectors(self, purchase_events):
        """
        Create purchase vectors for each user
        
        Parameters:
        -----------
        purchase_events : pd.DataFrame
            DataFrame containing purchase events
        """
        # Group purchases by user
        user_purchases = defaultdict(set)
        for _, row in purchase_events.iterrows():
            user_purchases[row['event_id']].add(row['item_id'])
            
        self.user_purchases = user_purchases
        return user_purchases
    
    def fit(self, purchase_events, sessions):
        """
        Fit the SKNN_E model
        
        Parameters:
        -----------
        purchase_events : pd.DataFrame
            DataFrame containing purchase events
        sessions : pd.DataFrame
            DataFrame containing session data
        """
        train_purchases, val_purchases, train_sessions, val_sessions = self.split_train_validation(
            purchase_events, sessions
        )
        
        self.val_purchases = val_purchases
        self.val_sessions = val_sessions

        train_purchases, train_sessions = self.preprocess_data(train_purchases, train_sessions)
        self.action_matrix = self.create_user_action_vectors(train_sessions)
        self.user_purchases = self.create_purchase_vectors(train_purchases)

        self.knn.fit(self.action_matrix, self.event_ids)
        
    def recommend(self, user_sessions, n_recommendations=5):
        """
        Generate recommendations for a user based on their sessions
        
        Parameters:
        -----------
        user_sessions : pd.DataFrame
            DataFrame containing sessions for the target user
        n_recommendations : int
            Number of items to recommend
            
        Returns:
        --------
        list
            List of recommended item IDs
        """
        # create action vector
        user_actions = set()
        for _, row in user_sessions.iterrows():
            action_feature = (f"{row['action_section']}_{row['action_type']}"
                            + (f"_{row['action_object']}" if pd.notna(row['action_object']) else ""))
            user_actions.add(action_feature)
        
        user_vector = self.mlb.transform([list(user_actions)])
        
        # find nearest neighbors
        distances, indices = self.knn.kneighbors(user_vector)
        neighbor_ids = [self.event_ids[idx] for idx in indices[0]]
        
        # find most common items
        neighbor_purchases = []
        for neighbor_id in neighbor_ids:
            if neighbor_id in self.user_purchases:
                neighbor_purchases.extend(self.user_purchases[neighbor_id])

        item_counts = defaultdict(int)
        for item in neighbor_purchases:
            item_counts[item] += 1
            
        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)

        recommendations = [item[0] for item in sorted_items[:n_recommendations]]
        return recommendations

# usage
if __name__ == "__main__":
    purchase_events = pd.read_csv('../Data Sets/purchase_events_train.csv')
    sessions = pd.read_csv('../Data Sets/sessions_train.csv')
    
    model = SKNN_E(n_neighbors=5, recent_sessions_window=30)
    model.fit(purchase_events, sessions)
    
    val_user_sessions = model.val_sessions[model.val_sessions['event_id'] == model.val_sessions['event_id'].iloc[0]]
    recommendations = model.recommend(val_user_sessions, n_recommendations=5)
    print("Recommended items:", recommendations)