# File: models/recommendation_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_ranking_model(input_dim: int):
    """
    Build a feed-forward neural network for ranking job relevance.
    input_dim: Dimension of the concatenated feature vector (user embedding, job embedding, and absolute difference).
    """
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Output relevance score between 0 and 1
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

if __name__ == "__main__":
    dummy_input_dim = 768 * 3  # Example for two BERT embeddings plus their absolute difference
    model = create_ranking_model(dummy_input_dim)
