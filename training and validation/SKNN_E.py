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

    def calculate_metrics(self, test_purchases, test_sessions, k=10):
        """
        Calculate HR@k, Precision@k, Recall@k, MRR@k, and MAP@k for test data
        
        Parameters:
        -----------
        test_purchases : pd.DataFrame
            DataFrame containing test purchase events
        test_sessions : pd.DataFrame
            DataFrame containing test session data
        k : int
            Number of recommendations to generate
            
        Returns:
        --------
        dict
            Dictionary containing the calculated metrics
        """
        metrics = {
            f'HR@{k}': 0,
            f'Precision@{k}': 0,
            f'Recall@{k}': 0,
            f'MRR@{k}': 0,
            f'MAP@{k}': 0
        }
        
        # Group test purchases by event_id
        test_purchase_dict = defaultdict(set)
        for _, row in test_purchases.iterrows():
            test_purchase_dict[row['event_id']].add(row['item_id'])
        
        # Get unique test users
        test_users = test_sessions['event_id'].unique()
        total_users = len(test_users)
        
        if total_users == 0:
            return metrics
        
        ap_sum = 0  # For MAP calculation
        
        for user_id in test_users:
            # Get user sessions and actual purchases
            user_sessions = test_sessions[test_sessions['event_id'] == user_id]
            actual_purchases = test_purchase_dict.get(user_id, set())
            
            if not actual_purchases:
                continue
                
            # Get recommendations for user
            recommendations = self.recommend(user_sessions, n_recommendations=k)
            recommended_set = set(recommendations)
            
            # Hit Ratio: 1 if at least one item is correctly recommended
            hit = len(actual_purchases & recommended_set) > 0
            metrics[f'HR@{k}'] += hit
            
            # Precision: proportion of recommended items that were actually purchased
            precision = len(actual_purchases & recommended_set) / len(recommendations) if recommendations else 0
            metrics[f'Precision@{k}'] += precision
            
            # Recall: proportion of purchased items that were recommended
            recall = len(actual_purchases & recommended_set) / len(actual_purchases) if actual_purchases else 0
            metrics[f'Recall@{k}'] += recall
            
            # MRR: reciprocal rank of the first relevant recommendation
            mrr = 0
            for i, item in enumerate(recommendations):
                if item in actual_purchases:
                    mrr = 1.0 / (i + 1)
                    break
            metrics[f'MRR@{k}'] += mrr
            
            # Average Precision for MAP
            ap = 0
            hits = 0
            for i, item in enumerate(recommendations):
                if item in actual_purchases:
                    hits += 1
                    ap += hits / (i + 1)
            if hits > 0:
                ap = ap / min(len(actual_purchases), k)
                ap_sum += ap
        
        # Normalize metrics by number of users
        for metric in ['HR', 'Precision', 'Recall', 'MRR']:
            metrics[f'{metric}@{k}'] /= total_users
        
        # Calculate MAP
        metrics[f'MAP@{k}'] = ap_sum / total_users
        
        return metrics

    def evaluate_test_set(self, test_purchases, test_sessions, k_values=[5, 10, 20]):
        """
        Evaluate the model on test data for multiple k values
        
        Parameters:
        -----------
        test_purchases : pd.DataFrame
            DataFrame containing test purchase events
        test_sessions : pd.DataFrame
            DataFrame containing test session data
        k_values : list
            List of k values to evaluate
            
        Returns:
        --------
        dict
            Dictionary containing metrics for each k value
        """
        results = {}
        
        for k in k_values:
            metrics = self.calculate_metrics(test_purchases, test_sessions, k)
            results[k] = metrics
            
            print(f"\nMetrics for k={k}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        
        return results

if __name__ == "__main__":
    purchase_events = pd.read_csv('../Data Sets/purchase_events_train.csv')
    sessions = pd.read_csv('../Data Sets/sessions_train.csv')
    
    model = SKNN_E(n_neighbors=5, recent_sessions_window=30)
    model.fit(purchase_events, sessions)
    
    val_user_sessions = model.val_sessions[model.val_sessions['event_id'] == model.val_sessions['event_id'].iloc[0]]
    recommendations = model.recommend(val_user_sessions, n_recommendations=5)
    print("Recommended items:", recommendations)

    test_purchases = pd.read_csv('../Data Sets/purchase_events_test.csv')
    test_sessions = pd.read_csv('../Data Sets/sessions_test.csv')

    results = model.evaluate_test_set(test_purchases, test_sessions, k_values=[3])
    print(results)