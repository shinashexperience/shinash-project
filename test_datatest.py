import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


try:
    import tensorflow as tf
    from tensorflow.keras import layers, Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow imported successfully")
except ImportError as e:
    print(f"❌ Error importing TensorFlow: {e}")
    print("⚠️  Falling back to basic implementation")
    TENSORFLOW_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
if TENSORFLOW_AVAILABLE:
    print("TensorFlow version:", tf.__version__)

# Load and prepare the data
# Using a publicly available dataset for healthcare costs
url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'
try:
    data = pd.read_csv(url)
    print("✅ Dataset loaded successfully")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    # Create sample data as fallback
    print("Creating sample data as fallback...")
    np.random.seed(42)
    n_samples = 1000
    data = pd.DataFrame({
        'age': np.random.randint(18, 65, n_samples),
        'sex': np.random.choice(['male', 'female'], n_samples),
        'bmi': np.random.uniform(18, 40, n_samples),
        'children': np.random.randint(0, 5, n_samples),
        'smoker': np.random.choice(['yes', 'no'], n_samples),
        'region': np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], n_samples),
        'expenses': np.random.uniform(1000, 50000, n_samples)
    })

# Display dataset info
print("Dataset shape:", data.shape)
print("\nColumn names:", data.columns.tolist())
print("\nFirst 5 rows:")
print(data.head())

# Periksa nama kolom target yang sebenarnya
possible_expense_columns = ['expenses', 'expense', 'charges', 'cost', 'costs', 'medical_cost']

# Cari kolom yang sesuai
expense_column = None
for col in possible_expense_columns:
    if col in data.columns:
        expense_column = col
        break

if expense_column is None:
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        expense_column = numeric_columns[-1]
        print(f"\n⚠️  Kolom 'expenses' tidak ditemukan. Menggunakan kolom numerik terakhir: '{expense_column}'")
    else:
        data['expenses'] = np.random.uniform(1000, 50000, len(data))
        expense_column = 'expenses'
        print(f"\n⚠️  Tidak ada kolom numerik. Membuat kolom 'expenses' baru")
else:
    print(f"\n✅ Kolom target ditemukan: '{expense_column}'")

# Convert categorical data to numbers
categorical_columns = ['sex', 'smoker', 'region']
available_categorical = [col for col in categorical_columns if col in data.columns]
print(f"Kolom kategorikal yang tersedia: {available_categorical}")

if available_categorical:
    data = pd.get_dummies(data, prefix=available_categorical, columns=available_categorical, drop_first=True)
else:
    print("⚠️  Tidak ada kolom kategorikal yang ditemukan")

print("\nData setelah konversi variabel kategorikal:")
print(data.head())
print("\nShape baru:", data.shape)

# Split the data into features and labels
features = data.copy()

if expense_column in features.columns:
    labels = features.pop(expense_column)
    print(f"\n✅ Berhasil memisahkan kolom target '{expense_column}'")
else:
    numeric_columns = features.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        expense_column = numeric_columns[-1]
        labels = features.pop(expense_column)
        print(f"\n⚠️  Kolom target tidak ditemukan. Menggunakan kolom numerik terakhir: '{expense_column}'")
    else:
        expense_column = features.columns[-1]
        labels = features.pop(expense_column)
        print(f"\n⚠️  Kolom target tidak ditemukan. Menggunakan kolom terakhir: '{expense_column}'")

print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")

# Split the data into training and test sets (80% train, 20% test)
train_dataset, test_dataset, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")

if not TENSORFLOW_AVAILABLE:
    print("\n" + "="*60)
    print("TENSORFLOW TIDAK TERSEDIA")
    print("Menjalankan analisis dengan metode alternatif...")
    print("="*60)
    
    # Analisis sederhana tanpa TensorFlow
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    # Scale features
    scaler = StandardScaler()
    train_dataset_scaled = scaler.fit_transform(train_dataset)
    test_dataset_scaled = scaler.transform(test_dataset)
    
    # Coba beberapa model
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    best_mae = float('inf')
    best_model_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(train_dataset_scaled, train_labels)
        predictions = model.predict(test_dataset_scaled)
        mae = mean_absolute_error(test_labels, predictions)
        mse = mean_squared_error(test_labels, predictions)
        r2 = r2_score(test_labels, predictions)
        
        print(f"{name} Results:")
        print(f"MAE: ${mae:.2f}")
        print(f"MSE: ${mse:.2f}")
        print(f"R²: {r2:.4f}")
        
        if mae < best_mae:
            best_mae = mae
            best_model_name = name
    
    # Plot results untuk model terbaik
    best_model = models[best_model_name]
    best_predictions = best_model.predict(test_dataset_scaled)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(test_labels, best_predictions, alpha=0.6, color='blue')
    plt.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'{best_model_name}\nPredictions vs Actual')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    error = best_predictions - test_labels
    plt.hist(error, bins=25, alpha=0.7, color='green')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Check challenge result
    if best_mae < 3500:
        print(f"\n🎉 CONGRATULATIONS! Challenge PASSED with {best_model_name}!")
        print(f"MAE: ${best_mae:.2f} (Target: < $3500)")
    else:
        print(f"\n💪 Keep trying! Challenge FAILED with {best_model_name}")
        print(f"MAE: ${best_mae:.2f} (Target: < $3500)")
    
else:
    # Normalize the data dengan TensorFlow
    normalizer = layers.Normalization()
    normalizer.adapt(np.array(train_dataset).astype('float32'))
    print("\nNormalization layer adapted successfully")

    # Build the model
    def build_model():
        model = Sequential([
            normalizer,
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_absolute_error',
            metrics=['mae', 'mse']
        )
        return model

    model = build_model()

    # Display model architecture
    print("\nModel architecture:")
    model.summary()

    # Train the model
    print("\nTraining the model...")
    
    class PrintProgress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: MAE = {logs['mae']:.2f}, Val MAE = {logs['val_mae']:.2f}")

    history = model.fit(
        train_dataset.astype('float32'),
        train_labels.astype('float32'),
        epochs=100,
        validation_split=0.2,
        verbose=0,
        batch_size=32,
        callbacks=[
            EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(patience=10, factor=0.5, verbose=1),
            PrintProgress()
        ]
    )

    print("Training completed!")

    # Evaluate the model
    print("\nEvaluating the model on test data...")
    test_results = model.evaluate(test_dataset.astype('float32'), test_labels.astype('float32'), verbose=0)

    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Test Loss (MAE): ${test_results[1]:.2f}")

    # Check if the model passes the challenge
    if test_results[1] < 3500:
        print("✅ Challenge PASSED - MAE is under $3500!")
        challenge_passed = True
    else:
        print(f"❌ Challenge FAILED - MAE is ${test_results[1]:.2f} (above $3500)")
        challenge_passed = False

    # Make predictions
    test_predictions = model.predict(test_dataset.astype('float32')).flatten()

    # Calculate additional metrics
    mae = mean_absolute_error(test_labels, test_predictions)
    mse = mean_squared_error(test_labels, test_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_labels, test_predictions)

    print(f"\nAdditional Metrics:")
    print(f"MAE: ${mae:.2f}")
    print(f"MSE: ${mse:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R² Score: {r2:.4f}")

    # Plot results
    plt.figure(figsize=(15, 10))

    # Plot 1: Predictions vs Actual
    plt.subplot(2, 2, 1)
    plt.scatter(test_labels, test_predictions, alpha=0.6, color='blue')
    plt.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], 'r--', lw=2)
    plt.xlabel('True Values (Expenses)')
    plt.ylabel('Predictions (Expenses)')
    plt.title('Predictions vs Actual Values')
    plt.grid(True, alpha=0.3)

    # Plot 2: Error distribution
    plt.subplot(2, 2, 2)
    error = test_predictions - test_labels
    plt.hist(error, bins=25, alpha=0.7, color='green')
    plt.xlabel('Prediction Error ($)')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)

    # Plot 3: Training history - MAE
    plt.subplot(2, 2, 3)
    if 'mae' in history.history:
        plt.plot(history.history['mae'], label='Training MAE', color='blue')
    if 'val_mae' in history.history:
        plt.plot(history.history['val_mae'], label='Validation MAE', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training History - MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: Training history - Loss
    plt.subplot(2, 2, 4)
    if 'loss' in history.history:
        plt.plot(history.history['loss'], label='Training Loss', color='blue')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History - Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Display some actual vs predicted comparisons
    comparison = pd.DataFrame({
        'Actual': test_labels.values,
        'Predicted': test_predictions,
        'Difference': np.abs(test_labels.values - test_predictions),
        'Error_Percentage': (np.abs(test_labels.values - test_predictions) / test_labels.values) * 100
    })

    print("\nSample predictions vs actual values (first 10 samples):")
    print(comparison.head(10))

    print(f"\nSummary Statistics for Errors:")
    print(f"Mean Absolute Error: ${comparison['Difference'].mean():.2f}")
    print(f"Max Error: ${comparison['Difference'].max():.2f}")
    print(f"Min Error: ${comparison['Difference'].min():.2f}")
    print(f"Mean Error Percentage: {comparison['Error_Percentage'].mean():.2f}%")

    print(f"\n{'='*50}")
    print("CHALLENGE RESULT")
    print(f"{'='*50}")
    if challenge_passed:
        print("🎉 CONGRATULATIONS! You passed the challenge!")
        print(f"Your model predicts healthcare costs within ${test_results[1]:.2f}")
    else:
        print("💪 Keep trying! You can improve the model.")
        print(f"Current MAE: ${test_results[1]:.2f} (Target: < $3500)")

print("\nScript execution completed!")