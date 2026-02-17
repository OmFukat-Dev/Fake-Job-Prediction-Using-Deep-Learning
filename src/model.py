from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from .config import CONFIG

def create_text_model(vocab_size, embedding_matrix=None):
    """Create LSTM model for text processing"""
    text_input = Input(shape=(CONFIG['model']['text_seq_length'],), name='text_input')
    
    if embedding_matrix is not None:
        embedding = Embedding(
            input_dim=vocab_size,
            output_dim=CONFIG['model']['embedding_dim'],
            weights=[embedding_matrix],
            trainable=False,
            name='embedding'
        )(text_input)
    else:
        embedding = Embedding(
            input_dim=vocab_size,
            output_dim=CONFIG['model']['embedding_dim'],
            name='embedding'
        )(text_input)
    
    lstm = LSTM(CONFIG['model']['lstm_units'], name='lstm')(embedding)
    dropout = Dropout(CONFIG['model']['dropout_rate'], name='dropout')(lstm)
    
    return text_input, dropout

def create_metadata_model(metadata_dim):
    """Create model for metadata features"""
    metadata_input = Input(shape=(metadata_dim,), name='metadata_input')
    dense = Dense(CONFIG['model']['dense_units'], activation='relu', name='metadata_dense')(metadata_input)
    dropout = Dropout(CONFIG['model']['dropout_rate'], name='metadata_dropout')(dense)
    
    return metadata_input, dropout

def create_combined_model(vocab_size, metadata_dim, embedding_matrix=None):
    """Create combined model with text and metadata"""
    # Text branch
    text_input, text_output = create_text_model(vocab_size, embedding_matrix)
    
    # Metadata branch
    metadata_input, metadata_output = create_metadata_model(metadata_dim)
    
    # Combine branches
    combined = Concatenate(name='concat')([text_output, metadata_output])
    dense1 = Dense(64, activation='relu', name='dense1')(combined)
    dropout1 = Dropout(0.3, name='final_dropout')(dense1)
    output = Dense(1, activation='sigmoid', name='output')(dropout1)
    
    # Create model
    model = Model(inputs=[text_input, metadata_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=CONFIG['training']['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model