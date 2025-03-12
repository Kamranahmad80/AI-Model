# File: models/recommendation_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_ranking_model(input_dim):
    """
    Build a simple feed-forward neural network to rank job relevance.
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
    # Example: suppose our concatenated vector is of length 3 * 768 (for two BERT embeddings and their absolute difference)
    dummy_input_dim = 768 * 3
    model = create_ranking_model(dummy_input_dim)
