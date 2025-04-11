import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns





st.set_page_config(page_title="ML Playground", layout="centered")
st.title("🧠 Machine Learning Playground")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df_encoded = pd.read_csv(uploaded_file)
    # Encode non-numeric columns
    df_encoded = df_encoded.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
     df_encoded[col] = df_encoded[col].astype('category').cat.codes

    st.write("🔍 Data Types")
    st.write(df_encoded.dtypes)

    st.write("🧪 Missing Values")
    st.write(df_encoded.isnull().sum())

    st.subheader("📄 Dataset Preview")
    st.dataframe(df_encoded.head())

    st.write("🔢 Shape:", df_encoded.shape)
    st.write("📋 Columns:", df_encoded.columns.tolist())
    
    st.subheader("🎯 Select Target Column")
    target_column = st.selectbox("Which column do you want to predict?", df_encoded.columns)

    st.subheader("🤖 Select ML Algorithm")
    model_name = st.selectbox(
        "Choose a model:",
        ["Linear Regression", "Decision Tree", "Random Forest"]
    )
    

    # Features = all columns except target
    X = df_encoded.drop(columns=[target_column])
    y = df_encoded[target_column]

    X = X.select_dtypes(include=['number'])

    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
   
    

    if X.empty:
        st.error("No numeric features to train on. Please upload a suitable dataset.")
    else:
     st.subheader("🧪 Train/Test Split")
     test_size = st.slider("Select test size (percentage of data used for testing):", 10, 50, 20, step=5)
     test_ratio = test_size / 100.0

        # Split the data
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)


     st.write("✅ Data is ready for training!")
 
        # Train the model based on selection
     if model_name == "Linear Regression":
      st.info("No tunable hyperparameters for Linear Regression.")
      model = LinearRegression()

     elif model_name == "Decision Tree":
      max_depth = st.slider("Max Depth", 1, 50, 10)
      min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
      model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)

     elif model_name == "Random Forest":
      n_estimators = st.slider("Number of Trees", 10, 500, 100, step=10)
      max_depth = st.slider("Max Depth", 1, 50, 10)
      model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
  
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)

     # Evaluate the model
     st.subheader("📊 Model Performance")
     st.write("**R² Score:**", round(r2_score(y_test, y_pred), 3))
     st.write("**Mean Squared Error:**", round(mean_squared_error(y_test, y_pred), 3))
          
     st.subheader("📈 Actual vs Predicted")

     fig, ax = plt.subplots()
     ax.plot(y_test.values, label="Actual", marker='o')
     ax.plot(y_pred, label="Predicted", marker='x')
     ax.set_xlabel("Sample")
     ax.set_ylabel(target_column)
     ax.legend()
     st.pyplot(fig)
    
    
     # Feature Importance Visualization
     if hasattr(model, 'feature_importances_'):
      st.subheader("📌 Feature Importance")
      importances = model.feature_importances_
      feature_names = X.columns
      importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
      importance_df = importance_df.sort_values(by='Importance', ascending=False)

      fig, ax = plt.subplots()
      ax.barh(importance_df['Feature'], importance_df['Importance'])
      ax.set_xlabel("Importance")
      ax.set_title("Feature Importance")
      st.pyplot(fig)
     else:
      st.info("ℹ️ Feature importance is not available for this model.")

     # Correlation Heatmap
     st.subheader("🧠 Feature Correlation Heatmap")
     # Only use numeric columns for correlation
     numeric_df = df_encoded.select_dtypes(include='number')

     fig, ax = plt.subplots(figsize=(10, 6))
     corr_matrix = numeric_df.corr()
     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
     st.pyplot(fig)



    
