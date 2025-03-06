import streamlit as st
import random
import smtplib
import spacy
import os
from dotenv import load_dotenv
import google.generativeai as genai
from email.message import EmailMessage
from langgraph.graph import StateGraph
from langchain.schema import SystemMessage, HumanMessage
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# Load environment variables
load_dotenv()
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
EMAIL_SENDER = st.secrets["EMAIL_ADDRESS"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Configure Google Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Store OTPs temporarily
otp_store = {}

# Function to extract location from user query
def extract_location(user_input):
    doc = nlp(user_input)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    return locations[0] if locations else None

# Function to send OTP via email
def send_otp(email):
    otp = str(random.randint(100000, 999999))
    otp_store[email] = otp  # Store OTP for verification

    msg = EmailMessage()
    msg.set_content(f"Your OTP for verification is: {otp}")
    msg["Subject"] = "Your OTP Code"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        return False

# Function to verify OTP
def verify_otp(email, entered_otp):
    return otp_store.get(email) == entered_otp

# Function to get historical monument details from Google Gemini
def get_monument_info(location):
    prompt = f"Tell me about the top historical monuments in {location} with their significance."
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text

# Define State for LangGraph
@dataclass
class ChatState:
    messages: List[Dict[str, Any]] = field(default_factory=list)
    email: Optional[str] = None
    otp_verified: bool = False
    location: Optional[str] = None

# LangGraph Workflow
workflow = StateGraph(ChatState)

# Step 1: Start Conversation
def greet_user(state: ChatState):
    state.messages.append({"role": "bot", "content": "Hello! I'm your historical monument guide. Where are you traveling?"})
    return state

# Step 2: Extract Location
def get_location(state: ChatState):
    user_input = state.messages[-1]["content"]
    location = extract_location(user_input)
    if location:
        state.location = location
        state.messages.append({"role": "bot", "content": f"Got it! You are interested in {location}. Can I have your email for verification?"})
    else:
        state.messages.append({"role": "bot", "content": "I couldn't detect a location. Can you mention it explicitly?"})
    return state

# Step 3: Request Email & Send OTP
def request_email(state: ChatState):
    user_input = state.messages[-1]["content"]
    if "@" in user_input:
        state.email = user_input
        success = send_otp(user_input)
        if success:
            state.messages.append({"role": "bot", "content": "OTP sent to your email. Please enter the OTP."})
        else:
            state.messages.append({"role": "bot", "content": "Failed to send OTP. Try again."})
    else:
        state.messages.append({"role": "bot", "content": "Please enter a valid email address."})
    return state

# Step 4: Verify OTP
def verify_user_otp(state: ChatState):
    user_input = state.messages[-1]["content"]
    if verify_otp(state.email, user_input):
        state.otp_verified = True
        state.messages.append({"role": "bot", "content": "OTP verified successfully! Fetching historical information..."})
    else:
        state.messages.append({"role": "bot", "content": "Invalid OTP. Try again."})
    return state

# Step 5: Fetch Historical Monument Information
def provide_info(state: ChatState):
    if state.otp_verified:
        monument_info = get_monument_info(state.location)
        state.messages.append({"role": "bot", "content": monument_info})
    else:
        state.messages.append({"role": "bot", "content": "OTP not verified. Cannot proceed."})
    return state

# Define Workflow Edges
workflow.add_node("greet", greet_user)
workflow.add_node("extract_location", get_location)
workflow.add_node("request_email", request_email)
workflow.add_node("verify_otp", verify_user_otp)
workflow.add_node("provide_info", provide_info)

workflow.set_entry_point("greet")
workflow.add_edge("greet", "extract_location")
workflow.add_edge("extract_location", "request_email")
workflow.add_edge("request_email", "verify_otp")
workflow.add_edge("verify_otp", "provide_info")

graph = workflow.compile()

# Streamlit UI
st.title("🗺️ Historical Monument Chatbot")

# ✅ FIX: Properly Initialize Session State
if "state" not in st.session_state:
    st.session_state["state"] = ChatState(messages=[])

user_input = st.text_input("You: ", "")

if user_input:
    # ✅ FIX: Ensure state is always ChatState object
    chat_state = st.session_state["state"]

    # ✅ FIX: Append user input properly
    chat_state.messages.append({"role": "user", "content": user_input})

    # ✅ FIX: Pass a ChatState instance to LangGraph
    new_state = graph.invoke(chat_state)

    # ✅ FIX: Store new state properly
    st.session_state["state"] = new_state

# ✅ FIX: Ensure state always has messages
if not hasattr(st.session_state["state"], "messages"):
    st.session_state["state"].messages = []

# Display Chat
for msg in st.session_state["state"].messages:
    if msg["role"] == "bot":
        st.write(f"🤖: {msg['content']}")
    else:
        st.write(f"🧑: {msg['content']}")


