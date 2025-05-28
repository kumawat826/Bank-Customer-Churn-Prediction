import streamlit as st
import pandas as pd
import pickle
import numpy as np
import openai
import utils as ut
import plotly.express as px


# ---------------------- LOGIN SYSTEM ----------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "users" not in st.session_state:
    st.session_state.users = {"admin": "admin123"}

def login_only():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.logged_in = True
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

if not st.session_state.logged_in:
    login_only()
    st.stop()

def logout():
    st.session_state.logged_in = False
    st.rerun()

# ---------------------- SIDEBAR ----------------------
st.sidebar.title("📊 Menu")
menu = st.sidebar.radio("Navigate", [
    "Home", 
    "Dataset Explorer", 
    "Churn Breakdown", 
    "Analytics Dashboard",
    "Profile", 
    "About", 
    "Contact / Feedback", 
    "Logout"
])

# ---------------------- GPT CLIENT ----------------------
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="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"
)

def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

xgboost_model = load_model("xgb_model.pkl")
random_forest_model = load_model("rf_model.pkl")
knn_model = load_model("knn_model.pkl")

def prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary):
    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCreditCard': has_credit_card,
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': estimated_salary,
        'Geography_France': 1 if location == 'France' else 0,
        'Geography_Germany': 1 if location == 'Germany' else 0,
        'Geography_Spain': 1 if location == 'Spain' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0,
    }
    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict

def make_predictions(input_df, input_dict):
    probabilities = {
        'XGBOOST': xgboost_model.predict_proba(input_df)[0][1],
        'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
        'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1],
    }
    avg_probability = np.mean(list(probabilities.values()))

    col1, col2 = st.columns(2)
    with col1:
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"The customer has a {avg_probability:.2%} probability of churning")
    with col2:
        fig_probs = ut.create_model_probablity_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)

    st.markdown("### Model Probabilities")
    for model, prob in probabilities.items():
        st.write(f"{model}: {prob:.2%}")
    st.write(f"Average Probability: {avg_probability:.2%}")

    return avg_probability

def explain_prediction(probability, input_dict, surname):
    prompt = f"""You are an expert data scientist...
    [Keep your existing explanation prompt here]
    """
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "system", "content": "You are an AI assistant."}, {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def generate_email(probability, input_dict, explanation, surname):
    prompt = f"""You are a manager at HS Bank...
    [Keep your existing email generation prompt here]
    """
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "system", "content": "You are an AI assistant."}, {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

df = pd.read_csv("churn.csv")

# ---------------------- PAGES ----------------------
if menu == "Home":
    st.title("Customer Churn Prediction")
    customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]
    selected_customer_option = st.selectbox("Select a customer", customers)

    if selected_customer_option:
        selected_customer_id = int(selected_customer_option.split("-")[0])
        selected_surname = selected_customer_option.split(" - ")[1]
        selected_customer = df.loc[df["CustomerId"] == selected_customer_id].iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            credit_score = st.number_input("Credit Score", 300, 800, int(selected_customer["CreditScore"]))
            location = st.selectbox("Location", ["Spain", "France", "Germany"], index=["Spain", "France", "Germany"].index(selected_customer["Geography"]))
            gender = st.radio("Gender", ["Male", "Female"], index=0 if selected_customer["Gender"] == "Male" else 1)
            age = st.number_input("Age", 18, 100, int(selected_customer["Age"]))
            tenure = st.number_input("Tenure (years)", 0, 50, int(selected_customer['Tenure']))
        with col2:
            balance = st.number_input("Balance", 0.0, value=float(selected_customer["Balance"]))
            num_products = st.number_input("Number of Products", 1, 10, int(selected_customer["NumOfProducts"]))
            has_credit_card = st.checkbox("Has Credit Card", value=bool(selected_customer["HasCrCard"]))
            is_active_member = st.checkbox("Is Active Member", value=bool(selected_customer["IsActiveMember"]))
            estimated_salary = st.number_input("Estimated Salary", 0.0, value=float(selected_customer["EstimatedSalary"]))

        input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary)
        avg_probability = make_predictions(input_df, input_dict)

        explanation = explain_prediction(avg_probability, input_dict, selected_surname)
        st.subheader("Explanation of Prediction")
        st.markdown(explanation)

        email = generate_email(avg_probability, input_dict, explanation, selected_surname)
        st.subheader("Personalized Email")
        st.markdown(email)

elif menu == "Dataset Explorer":
    st.title("📂 Dataset Explorer")
    st.dataframe(df)

    st.markdown("### Filter Dataset")
    country_filter = st.multiselect("Geography", df["Geography"].unique(), default=list(df["Geography"].unique()))
    gender_filter = st.multiselect("Gender", df["Gender"].unique(), default=list(df["Gender"].unique()))
    filtered_df = df[(df["Geography"].isin(country_filter)) & (df["Gender"].isin(gender_filter))]
    st.dataframe(filtered_df)

elif menu == "Churn Breakdown":
    st.title("📉 Churn Breakdown")

    churn_rate = df["Exited"].value_counts(normalize=True) * 100
    fig_pie = px.pie(names=["Stayed", "Churned"], values=churn_rate, title="Churn Rate")
    st.plotly_chart(fig_pie)

    st.markdown("### Churn by Geography")
    fig_geo = px.histogram(df, x="Geography", color="Exited", barmode="group", title="Churn by Country")
    st.plotly_chart(fig_geo)

    st.markdown("### Churn by Gender")
    fig_gender = px.histogram(df, x="Gender", color="Exited", barmode="group", title="Churn by Gender")
    st.plotly_chart(fig_gender)

elif menu == "Analytics Dashboard":
    st.title("📊 Analytics Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Churn Rate (%)", f"{df['Exited'].mean() * 100:.2f}")
        st.metric("Avg Credit Score", f"{df['CreditScore'].mean():.0f}")
    with col2:
        st.metric("Avg Age", f"{df['Age'].mean():.1f}")
        st.metric("Avg Balance", f"${df['Balance'].mean():,.2f}")

    st.markdown("### Age vs Balance")
    fig_scatter = px.scatter(df, x="Age", y="Balance", color="Exited", title="Age vs Balance (by Churn)")
    st.plotly_chart(fig_scatter)

elif menu == "Profile":
    st.title("👤 User Profile")

    st.markdown("""
    ### 🔑 Logged In As:
    - *Username:* {}

    ### 🧾 Session Details:
    - *Session Status:* Logged In
    - *Model Access:* XGBoost, Random Forest, KNN
    - *AI Services:* Enabled (Groq API via LLaMA 3)
    
    ### 💡 What You Can Do:
    - Predict customer churn with multiple models
    - Understand churn reasons with AI explanations
    - Generate personalized emails for at-risk customers
    - Send feedback via contact form

    ---
    ⚠ For security, always logout after use.
    """.format(list(st.session_state.users.keys())[0]))



elif menu == "About":
    st.title("ℹ About This App")

    st.markdown("""
    *Bank Customer Churn Prediction* is a machine learning-powered web application designed to help banks and financial institutions predict the likelihood of a customer leaving (churning) their services.

    This app uses pre-trained machine learning models (XGBoost, Random Forest, K-Nearest Neighbors) to analyze customer data and estimate the probability of churn. The results are visualized with model comparisons and AI-generated explanations and recommendations.

    ### 🔍 Input Features Explained

    - *Credit Score*  
      A numerical measure of a customer’s creditworthiness. Lower scores may indicate higher risk and higher likelihood of churn.

    - *Country (Geography)*  
      Indicates the customer's country (France, Germany, Spain). Geographic location may impact churn behavior due to regional policies or services.

    - *Gender*  
      The customer’s gender (Male or Female). Used to analyze demographic trends in churn behavior.

    - *Age*  
      Age of the customer. Younger or older customers may have varying levels of engagement with bank services.

    - *Tenure*  
      Number of years the customer has stayed with the bank. Longer tenure often implies loyalty, reducing churn chances.

    - *Account Balance*  
      Total balance in the customer’s account. Customers with low or zero balances may be more likely to churn.

    - *Estimated Salary*  
      Annual income estimate. It helps assess financial stability and segment customers accordingly.

    - *Number of Products*  
      Total financial products the customer uses (e.g., loans, credit cards). More products typically indicate stronger customer retention.

    - *Has Credit Card*  
      Indicates if the customer owns a credit card issued by the bank. Credit card holders may have deeper relationships with the bank.

    - *Is Active Member*  
      Whether the customer is actively engaging with bank services. Inactive customers tend to be more at risk of leaving.

    This tool empowers analysts and managers with AI-enhanced insights for proactive customer retention strategies.
    """)


elif menu == "Contact / Feedback":
    st.title("📨 Contact / Feedback")
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Message")

    if st.button("Send"):
        if name and email and message:
            st.success("Thanks for your feedback!")
        else:
            st.error("Please fill all fields before submitting.")


elif menu == "Logout":
    logout()