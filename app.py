import requests
import os
import io
import base64
import PIL
from PIL import Image
import numpy as np
import streamlit as st
import cohere
import json
import google.generativeai as genai
import fitz  # PyMuPDF
import time
import hashlib
import matplotlib.pyplot as plt
import glob
from datetime import datetime, timedelta
import pandas as pd
from functools import lru_cache
import threading
from queue import Queue
import concurrent.futures
import seaborn as sns
from prompt_engineering import (
    get_french_system_message,
    get_french_search_prompt,
    get_french_query_expansion_prompt,
    get_french_document_description_prompt,
    get_french_intent_prompt,
    FRENCH_INTENT_CATEGORIES
)
import shutil

# --- Constants for Persistence ---
EMBEDDINGS_FILE = "persistent_embeddings.npy"
PATHS_FILE = "persistent_paths.json"
CHAT_HISTORY_FILE = "chat_history.json"
ADMIN_CONFIG_PATH = "admin_config.json"
KEY_FILE = "key.txt"
ANALYTICS_FILE = "analytics_data.json"  # New file for analytics data
WORKSPACES_FILE = "workspaces.json"


def load_workspaces():
    if not os.path.exists(WORKSPACES_FILE):
        # On first run, create HR workspace with current data
        workspaces = {
            "HR": {
                "embeddings_file": EMBEDDINGS_FILE,
                "paths_file": PATHS_FILE,
                "chat_history_file": CHAT_HISTORY_FILE
            }
        }
        with open(WORKSPACES_FILE, 'w') as f:
            json.dump(workspaces, f, indent=2)
        return workspaces
    with open(WORKSPACES_FILE, 'r') as f:
        return json.load(f)

def save_workspaces(workspaces):
    with open(WORKSPACES_FILE, 'w') as f:
        json.dump(workspaces, f, indent=2)

def get_active_workspace():
    return st.session_state.get('active_workspace', 'HR')

def set_active_workspace(name):
    st.session_state.active_workspace = name
    # Load workspace data into session state
    workspaces = load_workspaces()
    ws = workspaces.get(name)
    if ws:
        # Load embeddings
        if os.path.exists(ws["embeddings_file"]) and os.path.exists(ws["paths_file"]):
            try:
                st.session_state.doc_embeddings = np.load(ws["embeddings_file"])
                with open(ws["paths_file"], 'r') as f:
                    st.session_state.image_paths = json.load(f)
            except Exception as e:
                st.session_state.doc_embeddings = None
                st.session_state.image_paths = []
        else:
            st.session_state.doc_embeddings = None
            st.session_state.image_paths = []
        # Load chat history
        if os.path.exists(ws["chat_history_file"]):
            try:
                with open(ws["chat_history_file"], 'r') as f:
                    st.session_state.history = json.load(f)
            except Exception:
                st.session_state.history = []
        else:
            st.session_state.history = []
    else:
        st.session_state.doc_embeddings = None
        st.session_state.image_paths = []
        st.session_state.history = []

def create_workspace(name):
    workspaces = load_workspaces()
    if name in workspaces:
        return False, "Workspace already exists."
    # Create empty files for new workspace
    emb_file = f"embeddings_{name}.npy"
    paths_file = f"paths_{name}.json"
    chat_file = f"chat_history_{name}.json"
    np.save(emb_file, np.empty((0, 512)))  # Assuming 512-dim embeddings
    with open(paths_file, 'w') as f:
        json.dump([], f)
    with open(chat_file, 'w') as f:
        json.dump([], f)
    workspaces[name] = {
        "embeddings_file": emb_file,
        "paths_file": paths_file,
        "chat_history_file": chat_file
    }
    save_workspaces(workspaces)
    return True, "Workspace created."

# --- Analytics Persistence Functions ---
def save_analytics_data():
    """Save analytics data to a JSON file"""
    try:
        analytics_data = {
            "analytics_data": st.session_state.analytics_data,
            "feedback_data": st.session_state.feedback_data
        }
        with open(ANALYTICS_FILE, 'w') as f:
            json.dump(analytics_data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving analytics data: {e}")
        return False

def load_analytics_data():
    """Load analytics data from JSON file"""
    try:
        if os.path.exists(ANALYTICS_FILE):
            with open(ANALYTICS_FILE, 'r') as f:
                data = json.load(f)
                return data.get("analytics_data", {}), data.get("feedback_data", {})
    except Exception as e:
        st.error(f"Error loading analytics data: {e}")
    return {}, {}

# --- Deduplicate feedback on load ---
def deduplicate_feedback(feedback_list):
    seen = set()
    deduped = []
    for entry in feedback_list:
        key = (entry.get("question"), entry.get("answer"))
        if key not in seen:
            deduped.append(entry)
            seen.add(key)
    return deduped

# --- Admin Utility Functions ---
def hash_password(password):
    """Create SHA-256 hash of password"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_credentials(username, password):
    """Verify admin credentials"""
    if not os.path.exists(ADMIN_CONFIG_PATH):
        # Create default admin config if it doesn't exist
        default_config = {
            "admin_username": "admin",
            "admin_password_hash": hash_password("admin"),
            "last_login": None
        }
        with open(ADMIN_CONFIG_PATH, 'w') as f:
            json.dump(default_config, f, indent=2)
    
    with open(ADMIN_CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    password_hash = hash_password(password)
    if (username == config.get('admin_username') and 
        password_hash == config.get('admin_password_hash')):
        # Update last login time
        config['last_login'] = datetime.now().isoformat()
        with open(ADMIN_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    
    return False

def change_password(new_password):
    """Change admin password"""
    if not os.path.exists(ADMIN_CONFIG_PATH):
        return False
    
    with open(ADMIN_CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    config['admin_password_hash'] = hash_password(new_password)
    
    with open(ADMIN_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    
    return True

def get_api_keys():
    """Get current API keys from key.txt"""
    if not os.path.exists(KEY_FILE):
        return {"cohere_key": "", "google_key": ""}
    
    with open(KEY_FILE, 'r') as f:
        lines = f.read().strip().split('\n')
    
    # Assuming first line is Cohere key, second line is Google key
    keys = {
        "cohere_key": lines[0] if len(lines) > 0 else "",
        "google_key": lines[1] if len(lines) > 1 else ""
    }
    
    return keys

def update_api_keys(cohere_key, google_key):
    """Update API keys in key.txt"""
    with open(KEY_FILE, 'w') as f:
        f.write(f"{cohere_key}\n{google_key}")
    
    return True

def get_document_stats():
    """Get statistics about indexed documents for the active workspace"""
    stats = {
        "total_pdf_pages": 0,
        "total_images": 0,
        "total_embeddings": 0,
        "pdf_documents": [],
        "embedding_size": 0,
        "pdf_details": {}  # New field to store PDF details
    }
    
    # Count PDF pages (all PDFs in pdf_pages/ are visible to all workspaces)
    pdf_page_folder = "pdf_pages"
    if os.path.exists(pdf_page_folder):
        pdf_folders = [f for f in os.listdir(pdf_page_folder) 
                      if os.path.isdir(os.path.join(pdf_page_folder, f))]
        
        for pdf_folder in pdf_folders:
            folder_path = os.path.join(pdf_page_folder, pdf_folder)
            page_files = glob.glob(os.path.join(folder_path, "*.png"))
            page_count = len(page_files)
            stats["total_pdf_pages"] += page_count
            stats["pdf_documents"].append({
                "name": pdf_folder,
                "pages": page_count
            })
            # Store PDF details including page images
            stats["pdf_details"][pdf_folder] = {
                "page_count": page_count,
                "page_images": sorted(page_files)  # Store sorted list of page images
            }
    
    # Count uploaded images (all images in uploaded_img/ are visible to all workspaces)
    upload_folder = "uploaded_img"
    if os.path.exists(upload_folder):
        stats["total_images"] = len(glob.glob(os.path.join(upload_folder, "*.png")))
    
    # Get embedding information for the active workspace
    workspaces = load_workspaces()
    ws = workspaces.get(get_active_workspace())
    if ws and os.path.exists(ws["embeddings_file"]):
        try:
            embeddings = np.load(ws["embeddings_file"], allow_pickle=True)
            stats["total_embeddings"] = len(embeddings)
            stats["embedding_size"] = os.path.getsize(ws["embeddings_file"]) / (1024 * 1024)  # Size in MB
        except:
            pass
    
    return stats

def generate_document_charts(stats):
    """Generate enhanced charts for document statistics"""
    charts = {}
    
    # Set Arial font for all plots
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
    # Document type distribution pie chart with enhanced styling
    if stats["total_pdf_pages"] > 0 or stats["total_images"] > 0:
        fig, ax = plt.subplots(figsize=(10, 7))
        labels = ['PDF Pages', 'Images']
        sizes = [stats["total_pdf_pages"], stats["total_images"]]
        colors = ['#FF9999', '#66B2FF']
        explode = (0.1, 0)  # explode the 1st slice (PDF Pages)
        
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90)
        
        # Enhance text properties
        plt.setp(autotexts, size=12, weight="bold")
        plt.setp(texts, size=12)
        
        ax.axis('equal')
        plt.title('Document Type Distribution', pad=20, fontsize=14, fontweight='bold')
        
        # Add legend with enhanced styling
        plt.legend(wedges, labels, title="Document Types", 
                  loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                  fontsize=12)
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
        buffer.seek(0)
        charts["document_distribution"] = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
    
    # PDF document bar chart with enhanced styling
    if stats["pdf_documents"]:
        fig, ax = plt.subplots(figsize=(12, 7))
        names = [doc["name"] for doc in stats["pdf_documents"]]
        pages = [doc["pages"] for doc in stats["pdf_documents"]]
        
        # Create gradient colors
        colors = plt.cm.Blues(np.linspace(0.5, 0.8, len(names)))
        
        bars = ax.bar(names, pages, color=colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.title('PDF Documents Page Count', pad=20, fontsize=14, fontweight='bold')
        plt.xlabel('Document Name', fontsize=12)
        plt.ylabel('Number of Pages', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
        buffer.seek(0)
        charts["pdf_distribution"] = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
    
    # Add new visualization: Document Size Distribution
    if stats["embedding_size"] > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        sizes = [stats["embedding_size"], 
                stats["total_pdf_pages"] * 0.1,  # Approximate size per PDF page
                stats["total_images"] * 0.2]     # Approximate size per image
        labels = ['Embeddings', 'PDF Pages', 'Images']
        colors = ['#FF9999', '#66B2FF', '#99FF99']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90)
        
        # Enhance text properties
        plt.setp(autotexts, size=12, weight="bold")
        plt.setp(texts, size=12)
        
        ax.axis('equal')
        plt.title('Storage Distribution', pad=20, fontsize=14, fontweight='bold')
        
        # Add legend with enhanced styling
        plt.legend(wedges, labels, title="Storage Types", 
                  loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                  fontsize=12)
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
        buffer.seek(0)
        charts["storage_distribution"] = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
    
    # Add new visualization: Document Growth Timeline
    if stats["pdf_documents"]:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort documents by name (assuming names contain dates or are chronological)
        sorted_docs = sorted(stats["pdf_documents"], key=lambda x: x["name"])
        names = [doc["name"] for doc in sorted_docs]
        pages = [doc["pages"] for doc in sorted_docs]
        
        # Create line plot with markers
        ax.plot(names, pages, marker='o', linestyle='-', linewidth=2, markersize=8)
        
        # Add data points
        for i, (name, page) in enumerate(zip(names, pages)):
            ax.text(i, page, f'{page}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.title('Document Growth Timeline', pad=20, fontsize=14, fontweight='bold')
        plt.xlabel('Document Name', fontsize=12)
        plt.ylabel('Number of Pages', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
        buffer.seek(0)
        charts["growth_timeline"] = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
    
    return charts

# --- Persistence Functions ---
def load_persistent_data():
    """Loads embeddings and image paths from the active workspace's files if they exist."""
    workspaces = load_workspaces()
    ws = workspaces.get(get_active_workspace())
    loaded_embeddings = None
    loaded_paths = []
    if ws and os.path.exists(ws["embeddings_file"]) and os.path.exists(ws["paths_file"]):
        try:
            loaded_embeddings = np.load(ws["embeddings_file"])
            with open(ws["paths_file"], 'r') as f:
                loaded_paths = json.load(f)
            # Validate that all paths still exist
            valid_paths = []
            valid_indices = []
            for i, path in enumerate(loaded_paths):
                if os.path.exists(path):
                    valid_paths.append(path)
                    valid_indices.append(i)
                else:
                    st.sidebar.warning(f"File not found: {path}")
            # If some paths are invalid, filter the embeddings
            if len(valid_paths) < len(loaded_paths):
                if len(valid_paths) == 0:
                    st.sidebar.error("No valid files found in saved data.")
                    return None, []
                loaded_embeddings = loaded_embeddings[valid_indices]
                loaded_paths = valid_paths
            # Basic validation
            if loaded_embeddings.shape[0] == len(loaded_paths):
                return loaded_embeddings, loaded_paths
            else:
                st.sidebar.warning("Mismatch between saved embeddings and paths. Ignoring saved data.")
                return None, []
        except Exception as e:
            st.sidebar.error(f"Error loading persistent data: {e}")
            return None, []
    return None, []

def save_persistent_data(embeddings, paths):
    """Saves embeddings and image paths to local files for the active workspace."""
    try:
        workspaces = load_workspaces()
        ws = workspaces.get(get_active_workspace())
        if ws:
            np.save(ws["embeddings_file"], embeddings)
            with open(ws["paths_file"], 'w') as f:
                json.dump(paths, f)
        return True
    except Exception as e:
        st.sidebar.error(f"Error saving persistent data: {e}")
        return False

def load_chat_history():
    """Loads chat history from local file for the active workspace."""
    try:
        workspaces = load_workspaces()
        ws = workspaces.get(get_active_workspace())
        if ws:
            with open(ws["chat_history_file"], 'r') as f:
                return json.load(f)
    except Exception as e:
        st.sidebar.error(f"Error loading chat history: {e}")
    return []

def save_chat_history(history):
    """Saves chat history to local file for the active workspace."""
    try:
        workspaces = load_workspaces()
        ws = workspaces.get(get_active_workspace())
        if ws:
            with open(ws["chat_history_file"], 'w') as f:
                json.dump(history, f)
        return True
    except Exception as e:
        st.sidebar.error(f"Error saving chat history: {e}")
        return False

# --- Chat Management Utilities ---
CHATS_DIR = "chats"
os.makedirs(CHATS_DIR, exist_ok=True)

def list_chat_files():
    return sorted(glob.glob(os.path.join(CHATS_DIR, "chat_history_*.json")), reverse=True)

def save_current_chat():
    if st.session_state.history:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chat_file = os.path.join(CHATS_DIR, f"chat_history_{timestamp}.json")
        with open(chat_file, "w") as f:
            json.dump(st.session_state.history, f, indent=2)

def load_chat_file(chat_file):
    try:
        with open(chat_file, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load chat: {e}")
        return []

# --- Initialize Session State ---
if 'image_paths' not in st.session_state:
    st.session_state.image_paths = []
if 'doc_embeddings' not in st.session_state:
    st.session_state.doc_embeddings = None
if 'last_uploaded_names' not in st.session_state:
    st.session_state.last_uploaded_names = []
if 'history' not in st.session_state:
    st.session_state.history = []  # Initialize with empty history
if 'show_references' not in st.session_state:
    st.session_state.show_references = {}
if 'question' not in st.session_state:
    st.session_state.question = ""
if 'show_admin' not in st.session_state:
    st.session_state.show_admin = False
if 'admin_authenticated' not in st.session_state:
    st.session_state.admin_authenticated = False
if 'admin_page' not in st.session_state:
    st.session_state.admin_page = "API Keys"
if 'temp_message' not in st.session_state:
    st.session_state.temp_message = None
if 'deleted_chat' not in st.session_state:
    st.session_state.deleted_chat = None
if 'processing_times' not in st.session_state:
    st.session_state.processing_times = {}
if 'api_usage' not in st.session_state:
    st.session_state.api_usage = {
        "cohere": {"calls": 0, "last_reset": datetime.now()},
        "google": {"calls": 0, "last_reset": datetime.now()},
        "detailed_calls": []  # New field for detailed API call tracking
    }
elif "detailed_calls" not in st.session_state.api_usage:
    # Initialize detailed_calls for existing sessions
    st.session_state.api_usage["detailed_calls"] = []
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = None
if 'zoom_level' not in st.session_state:
    st.session_state.zoom_level = 1.0

# Load saved analytics data
saved_analytics, saved_feedback = load_analytics_data()
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = saved_feedback or {
        "search_feedback": [],
        "answer_feedback": [],
        "performance_metrics": {
            "search_times": [],
            "answer_times": [],
            "query_success": []
        }
    }
# Deduplicate answer_feedback on load
st.session_state.feedback_data["answer_feedback"] = deduplicate_feedback(st.session_state.feedback_data.get("answer_feedback", []))
if 'analytics_data' not in st.session_state:
    st.session_state.analytics_data = saved_analytics or {
        "query_patterns": {},
        "intent_distribution": {},
        "success_rates": {},
        "user_engagement": {
            "total_queries": 0,
            "successful_queries": 0,
            "feedback_count": 0
        }
    }

# --- Initialize API Clients ---
co = None
try:
    # Load API keys
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, 'r') as f:
            lines = f.read().strip().split('\n')
        
        cohere_api_key = lines[0] if len(lines) > 0 else ""
        google_api_key = lines[1] if len(lines) > 1 else ""
        
        # Initialize clients with proper error handling
        if cohere_api_key:
            try:
                co = cohere.Client(api_key=cohere_api_key)
            except Exception as e:
                error_placeholder = st.sidebar.empty()
                error_placeholder.error(f"Cohere Initialization Failed: {e}")
                time.sleep(5)
                error_placeholder.empty()
        
        if google_api_key:
            try:
                genai.configure(api_key=google_api_key)
            except Exception as e:
                error_placeholder = st.sidebar.empty()
                error_placeholder.error(f"Gemini Configuration Failed: {e}")
                time.sleep(5)
                error_placeholder.empty()
except Exception as e:
    st.error(f"Error initializing API clients: {str(e)}")

# --- Streamlit App Configuration ---
st.set_page_config(layout="wide", page_title="Chatbot Évry")

# Load custom CSS from a separate file if it exists
css_path = os.path.join(os.path.dirname(__file__), "custom_styles.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Sidebar Redesign ---
with st.sidebar:
    # University logo at the very top, reduced size
    st.markdown("<div style='margin-top: -1.5rem; margin-bottom: 0.5rem; text-align: center;'>", unsafe_allow_html=True)
    st.image("logo.gif", use_container_width=True, output_format="GIF")
    st.markdown("</div>", unsafe_allow_html=True)

    # + NEW CHAT button with enhanced styling
    if st.button("+ NEW CHAT", key="new_chat_sidebar", use_container_width=True):
        save_current_chat()
        st.session_state.history = []
        st.rerun()

    # Previous chats with enhanced styling and functionality
    st.markdown('<div class="sidebar-header">Chat History</div>', unsafe_allow_html=True)
    
    # Display temporary message if exists
    if st.session_state.temp_message:
        if st.session_state.temp_message["type"] == "success":
            st.success(st.session_state.temp_message["text"])
        elif st.session_state.temp_message["type"] == "error":
            st.error(st.session_state.temp_message["text"])
        # Clear the message after displaying it
        st.session_state.temp_message = None
    
    chat_files = list_chat_files()
    
    for chat_file in chat_files:
        try:
            with open(chat_file, "r") as f:
                chat_data = json.load(f)
            chat_title = None
            for msg in chat_data:
                if msg.get("role") == "user" and msg.get("content"):
                    chat_title = msg["content"][:40] + ("..." if len(msg["content"]) > 40 else "")
                    break
            if not chat_title:
                chat_title = os.path.basename(chat_file).replace("chat_history_", "").replace(".json", "")
        except Exception:
            chat_title = os.path.basename(chat_file).replace("chat_history_", "").replace(".json", "")
        
        # Create a container for each chat item with hover functionality
        st.markdown(f"""
        <div class="chat-history-item" onclick="document.querySelector('#load_{chat_title}_{chat_file}').click()">
            <span class="chat-title">{chat_title}</span>
            <div class="chat-actions">
                <button class="chat-action-button delete-button" onclick="event.stopPropagation(); document.querySelector('#delete_{chat_title}_{chat_file}').click();">🗑️</button>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Hidden buttons for functionality
        col1, col2 = st.columns([1, 1])
        with col1:
            st.button("Load", key=f"load_{chat_title}_{chat_file}", type="primary", use_container_width=True, on_click=lambda: load_chat(chat_file))
        with col2:
            st.button("Delete", key=f"delete_{chat_title}_{chat_file}", type="secondary", use_container_width=True, on_click=lambda: delete_chat(chat_file))

    # Admin section at the bottom
    st.markdown('<div class="admin-section">', unsafe_allow_html=True)
    
    # Admin Panel button with enhanced styling
    if st.button("🔐 Admin Panel" if not st.session_state.show_admin else "🔒 Close Admin Panel", 
                 key="admin_panel_sidebar", 
                 use_container_width=True):
        st.session_state.show_admin = not st.session_state.show_admin
        if not st.session_state.show_admin:
            st.session_state.admin_authenticated = False
        st.rerun()
    
    # Load Previous Data button with enhanced styling
    if st.button("Load Data", 
                 key="load_previous_data_sidebar", 
                 use_container_width=True):
        if co:
            with st.spinner("Loading previous data..."):
                loaded_embeddings, loaded_paths = load_persistent_data()
                if loaded_embeddings is not None and len(loaded_paths) > 0:
                    st.session_state.doc_embeddings = loaded_embeddings
                    st.session_state.image_paths = loaded_paths
                    # Clear chat history when loading new data
                    st.session_state.history = []
                    st.success(f"Successfully loaded {len(loaded_paths)} items from previous session for workspace {get_active_workspace()}.")
                    st.rerun()
                else:
                    st.error(f"No previous data found or data could not be loaded for workspace {get_active_workspace()}.")
        else:
            st.warning("Please enter API keys first to load previous data.")
    
    # Workspace button below Load Data
    if st.button("Workspace", key="workspace_sidebar", use_container_width=True):
        st.session_state.show_workspace = not st.session_state.get('show_workspace', False)
        if not st.session_state.show_workspace:
            st.session_state.workspace_panel_opened = False
        st.rerun()
    
    # Show active workspace
    active_ws = get_active_workspace()
    st.markdown(f'<div style="margin-bottom:0.5rem;"><b>Active Workspace:</b> <span style="color:#6366F1;">{active_ws}</span></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Add helper functions for chat operations
def load_chat(chat_file):
    """Load a specific chat file and clear current chat history"""
    st.session_state.history = load_chat_file(chat_file)
    st.rerun()

def delete_chat(chat_file):
    """Delete a chat file and clear current chat history"""
    try:
        os.remove(chat_file)
        # Clear current chat history
        st.session_state.history = []
        # Store the success message in session state
        st.session_state.temp_message = {"type": "success", "text": "Chat deleted successfully!"}
        st.rerun()
    except Exception as e:
        st.session_state.temp_message = {"type": "error", "text": f"Failed to delete chat: {e}"}

# --- Helper functions ---
max_pixels = 1568*1568  # Max resolution for images

def resize_image(pil_image: PIL.Image.Image) -> None:
    """Resizes the image in-place if it exceeds max_pixels."""
    org_width, org_height = pil_image.size
    if org_width * org_height > max_pixels:
        scale_factor = (max_pixels / (org_width * org_height)) ** 0.5
        new_width = int(org_width * scale_factor)
        new_height = int(org_height * scale_factor)
        pil_image.thumbnail((new_width, new_height))

def base64_from_image(img_path: str) -> str:
    """Converts an image file to a base64 encoded string."""
    pil_image = PIL.Image.open(img_path)
    img_format = pil_image.format if pil_image.format else "PNG"
    resize_image(pil_image)
    with io.BytesIO() as img_buffer:
        pil_image.save(img_buffer, format=img_format)
        img_buffer.seek(0)
        img_data = f"data:image/{img_format.lower()};base64,"+base64.b64encode(img_buffer.read()).decode("utf-8")
    return img_data

def pil_to_base64(pil_image: PIL.Image.Image) -> str:
    """Converts a PIL image to a base64 encoded string."""
    img_format = pil_image.format if pil_image.format else "PNG"
    resize_image(pil_image)
    with io.BytesIO() as img_buffer:
        pil_image.save(img_buffer, format=img_format)
        img_buffer.seek(0)
        img_data = f"data:image/{img_format.lower()};base64,"+base64.b64encode(img_buffer.read()).decode("utf-8")
    return img_data

@lru_cache(maxsize=100)
def load_image_cached(image_path):
    """Cache image loading to improve performance"""
    return Image.open(image_path)

def track_api_usage(api_name, operation_type=None, success=True, response_time=None):
    """Enhanced API usage tracking with detailed metrics"""
    current_time = datetime.now()
    
    # Reset daily counters
    if current_time - st.session_state.api_usage[api_name]["last_reset"] > timedelta(days=1):
        st.session_state.api_usage[api_name]["calls"] = 0
        st.session_state.api_usage[api_name]["last_reset"] = current_time
    
    # Increment basic counter
    st.session_state.api_usage[api_name]["calls"] += 1
    
    # Add detailed call information
    call_info = {
        "timestamp": current_time.isoformat(),
        "api": api_name,
        "operation": operation_type or "unknown",
        "success": success,
        "response_time": response_time
    }
    st.session_state.api_usage["detailed_calls"].append(call_info)
    
    # Save analytics data after updating
    save_analytics_data()

def process_pdf_async(pdf_file, cohere_client, base_output_folder="pdf_pages"):
    """Asynchronous PDF processing with progress tracking"""
    start_time = time.time()
    pdf_filename = pdf_file.name
    output_folder = os.path.join(base_output_folder, os.path.splitext(pdf_filename)[0])
    os.makedirs(output_folder, exist_ok=True)
    
    # Create a queue for progress updates
    progress_queue = Queue()
    
    def process_pages():
        try:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            total_pages = len(doc)
            
            for i, page in enumerate(doc.pages()):
                page_num = i + 1
                page_img_path = os.path.join(output_folder, f"page_{page_num}.png")
                
                # Render page to pixmap
                pix = page.get_pixmap(dpi=150)
                pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pil_image.save(page_img_path, "PNG")
                
                # Update progress
                progress_queue.put((i + 1) / total_pages)
            
            doc.close()
            progress_queue.put(1.0)  # Signal completion
            
        except Exception as e:
            progress_queue.put(("error", str(e)))
    
    # Start processing in a separate thread
    thread = threading.Thread(target=process_pages)
    thread.start()
    
    return progress_queue, output_folder

def search_within_pdf(pdf_name, query, co_client):
    """Search for text within a PDF's pages using advanced prompt engineering"""
    pdf_folder = os.path.join("pdf_pages", pdf_name)
    if not os.path.exists(pdf_folder):
        return []
    
    results = []
    page_files = sorted(glob.glob(os.path.join(pdf_folder, "*.png")))
    
    for page_file in page_files:
        try:
            start_time = time.time()
            img_data = base64_from_image(page_file)
            model = genai.GenerativeModel('gemini-pro-vision')
            
            # Use French-optimized search prompt
            search_prompt = get_french_search_prompt(query)
            
            response = model.generate_content([
                search_prompt,
                {"mime_type": "image/jpeg", "data": img_data.split(",")[1]}
            ])
            response_time = time.time() - start_time
            track_api_usage("google", operation_type="vision_search", success=True, response_time=response_time)
            
            try:
                response_data = json.loads(response.text)
                if response_data.get("relevant", False) and response_data.get("confidence", 0) > 0.6:
                    page_num = os.path.basename(page_file).replace("page_", "").replace(".png", "")
                    results.append({
                        "path": page_file, 
                        "page_num": page_num,
                        "confidence": response_data.get("confidence", 0),
                        "matches": response_data.get("matches", []),
                        "explanation": response_data.get("explanation", ""),
                        "context": response_data.get("context", "")
                    })
            except json.JSONDecodeError:
                if "pertinent" in response.text.lower() and ("true" in response.text.lower() or "oui" in response.text.lower()):
                    page_num = os.path.basename(page_file).replace("page_", "").replace(".png", "")
                    results.append({
                        "path": page_file, 
                        "page_num": page_num,
                        "confidence": 0.7,
                        "matches": [],
                        "explanation": "Contenu pertinent détecté (détails non disponibles)",
                        "context": "Contexte administratif français"
                    })
        except Exception as e:
            track_api_usage("google", operation_type="vision_search", success=False)
            st.error(f"Error searching page {page_file}: {e}")
    
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return [(r["path"], r["page_num"]) for r in results]

def display_document_statistics():
    """Display enhanced document statistics in the admin panel"""
    st.header("Document Statistics")
    
    # Get document statistics
    stats = get_document_stats()
    
    # Create tabs for different sections
    stat_tabs = st.tabs(["Overview", "PDF Documents", "Analytics", "API Usage", "System Analytics"])
    
    # Overview tab
    with stat_tabs[0]:
        # Create three columns for key metrics with enhanced styling
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Total PDF Pages</div>
            </div>
            """.format(stats["total_pdf_pages"]), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Total Images</div>
            </div>
            """.format(stats["total_images"]), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Total Embeddings</div>
            </div>
            """.format(stats["total_embeddings"]), unsafe_allow_html=True)
        
        # Display embedding size if available
        if stats["embedding_size"] > 0:
            st.markdown("""
            <div class="metric-card" style="max-width: 300px; margin: 1rem auto;">
                <div class="metric-value">{:.2f} MB</div>
                <div class="metric-label">Embeddings File Size</div>
            </div>
            """.format(stats["embedding_size"]), unsafe_allow_html=True)
        
        # Generate and display charts
        charts = generate_document_charts(stats)
        
        if charts:
            st.subheader("Document Visualizations")
            
            # Create tabs for different visualizations
            viz_tabs = st.tabs(["Document Distribution", "PDF Analysis", "Storage Analysis", "Growth Timeline"])
            
            # Document Distribution tab
            with viz_tabs[0]:
                if "document_distribution" in charts:
                    st.image(
                        f"data:image/png;base64,{charts['document_distribution']}",
                        caption="Document Type Distribution",
                        use_container_width=True
                    )
            
            # PDF Analysis tab
            with viz_tabs[1]:
                if "pdf_distribution" in charts:
                    st.image(
                        f"data:image/png;base64,{charts['pdf_distribution']}",
                        caption="PDF Documents Page Count",
                        use_container_width=True
                    )
            
            # Storage Analysis tab
            with viz_tabs[2]:
                if "storage_distribution" in charts:
                    st.image(
                        f"data:image/png;base64,{charts['storage_distribution']}",
                        caption="Storage Distribution",
                        use_container_width=True
                    )
            
            # Growth Timeline tab
            with viz_tabs[3]:
                if "growth_timeline" in charts:
                    st.image(
                        f"data:image/png;base64,{charts['growth_timeline']}",
                        caption="Document Growth Timeline",
                        use_container_width=True
                    )
    
    # PDF Documents tab
    with stat_tabs[1]:
        if stats["pdf_documents"]:
            # PDF selection
            pdf_names = [doc["name"] for doc in stats["pdf_documents"]]
            selected_pdf = st.selectbox(
                "Select a PDF document to view its pages:",
                pdf_names,
                format_func=lambda x: f"{x} ({stats['pdf_details'][x]['page_count']} pages)"
            )
            
            if selected_pdf:
                # PDF viewer
                st.markdown(f"### {selected_pdf}")
                st.markdown(f"**Total Pages:** {stats['pdf_details'][selected_pdf]['page_count']}")
                
                # Display pages
                page_images = stats['pdf_details'][selected_pdf]['page_images']
                cols = st.columns(3)
                
                for i, img_path in enumerate(page_images):
                    with cols[i % 3]:
                        page_num = os.path.basename(img_path).replace("page_", "").replace(".png", "")
                        img = load_image_cached(img_path)
                        st.image(img, caption=f"Page {page_num}", use_container_width=True)
        else:
            st.info("No PDF documents have been processed yet.")
    
    # Analytics tab
    with stat_tabs[2]:
        st.subheader("Processing Statistics")
        
        # Display processing times
        if st.session_state.processing_times:
            processing_data = pd.DataFrame([
                {"Document": k, "Time (s)": v} 
                for k, v in st.session_state.processing_times.items()
            ])
            st.bar_chart(processing_data.set_index("Document"))
        
        # Display document statistics
        st.subheader("Document Statistics")
        doc_stats = pd.DataFrame([
            {"Metric": "Total PDFs", "Value": len(stats["pdf_documents"])},
            {"Metric": "Total Pages", "Value": stats["total_pdf_pages"]},
            {"Metric": "Total Images", "Value": stats["total_images"]},
            {"Metric": "Total Embeddings", "Value": stats["total_embeddings"]}
        ])
        st.dataframe(doc_stats)
    
    # API Usage tab
    with stat_tabs[3]:
        st.markdown('<div class="tab-header">📊 API Usage Statistics</div>', unsafe_allow_html=True)
        
        # Create columns for key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{st.session_state.api_usage["cohere"]["calls"]}</div>
                <div class="metric-label">Cohere API Calls</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{st.session_state.api_usage["google"]["calls"]}</div>
                <div class="metric-label">Google API Calls</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_calls = st.session_state.api_usage["cohere"]["calls"] + st.session_state.api_usage["google"]["calls"]
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_calls}</div>
                <div class="metric-label">Total API Calls</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Create sub-tabs for different API visualizations
        api_tabs = st.tabs(["Usage Over Time", "Operation Types", "Response Times", "Success Rates"])
        
        # Ensure detailed_calls exists
        if "detailed_calls" not in st.session_state.api_usage:
            st.session_state.api_usage["detailed_calls"] = []
        
        # Usage Over Time tab
        with api_tabs[0]:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            if st.session_state.api_usage["detailed_calls"]:
                # Convert timestamps to datetime and create DataFrame
                df_calls = pd.DataFrame(st.session_state.api_usage["detailed_calls"])
                df_calls['timestamp'] = pd.to_datetime(df_calls['timestamp'])
                df_calls['date'] = df_calls['timestamp'].dt.date
                
                # Group by date and API
                daily_calls = df_calls.groupby(['date', 'api']).size().unstack(fill_value=0)
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                daily_calls.plot(kind='line', marker='o', ax=ax)
                ax.set_title('API Calls Over Time', fontsize=14, fontweight='bold')
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Number of Calls', fontsize=12)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No API call data available yet.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Operation Types tab
        with api_tabs[1]:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            if st.session_state.api_usage["detailed_calls"]:
                df_calls = pd.DataFrame(st.session_state.api_usage["detailed_calls"])
                operation_counts = df_calls.groupby(['api', 'operation']).size().unstack(fill_value=0)
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                operation_counts.plot(kind='bar', ax=ax)
                ax.set_title('API Calls by Operation Type', fontsize=14, fontweight='bold')
                ax.set_xlabel('API', fontsize=12)
                ax.set_ylabel('Number of Calls', fontsize=12)
                plt.xticks(rotation=45)
                plt.legend(title='Operation Type', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No API call data available yet.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Response Times tab
        with api_tabs[2]:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            if st.session_state.api_usage["detailed_calls"]:
                df_calls = pd.DataFrame(st.session_state.api_usage["detailed_calls"])
                df_calls = df_calls[df_calls['response_time'].notna()]
                
                if not df_calls.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(data=df_calls, x='api', y='response_time', ax=ax)
                    ax.set_title('API Response Time Distribution', fontsize=14, fontweight='bold')
                    ax.set_xlabel('API', fontsize=12)
                    ax.set_ylabel('Response Time (seconds)', fontsize=12)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No response time data available yet.")
            else:
                st.info("No API call data available yet.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Success Rates tab
        with api_tabs[3]:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            if st.session_state.api_usage["detailed_calls"]:
                df_calls = pd.DataFrame(st.session_state.api_usage["detailed_calls"])
                success_rates = df_calls.groupby(['api', 'operation'])['success'].mean().unstack(fill_value=0)
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                success_rates.plot(kind='bar', ax=ax)
                ax.set_title('API Success Rates by Operation', fontsize=14, fontweight='bold')
                ax.set_xlabel('API', fontsize=12)
                ax.set_ylabel('Success Rate', fontsize=12)
                plt.xticks(rotation=45)
                plt.legend(title='Operation Type', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No API call data available yet.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Reset counters button
        if st.button("Reset API Counters"):
            st.session_state.api_usage = {
                "cohere": {"calls": 0, "last_reset": datetime.now()},
                "google": {"calls": 0, "last_reset": datetime.now()},
                "detailed_calls": []
            }
            st.success("API counters reset successfully!")

    # Add System Analytics tab
    with stat_tabs[4]:
        display_analytics_dashboard()

def process_pdf_file(pdf_file, cohere_client, base_output_folder="pdf_pages"):
    """Process PDF file with progress tracking and async processing"""
    start_time = time.time()
    progress_queue, output_folder = process_pdf_async(pdf_file, cohere_client, base_output_folder)
    
    # Show progress bar
    progress_bar = st.progress(0.0)
    while True:
        progress = progress_queue.get()
        if isinstance(progress, tuple) and progress[0] == "error":
            st.error(f"Error processing PDF: {progress[1]}")
            return [], None
        progress_bar.progress(progress)
        if progress >= 1.0:
            break
    
    progress_bar.empty()
    
    # Process the generated images
    page_image_paths = []
    page_embeddings = []
    
    page_files = sorted(glob.glob(os.path.join(output_folder, "*.png")))
    for page_file in page_files:
        page_image_paths.append(page_file)
        base64_img = base64_from_image(page_file)
        emb = compute_image_embedding(base64_img, _cohere_client=cohere_client)
        if emb is not None:
            page_embeddings.append(emb)
    
    # Track processing time
    processing_time = time.time() - start_time
    st.session_state.processing_times[pdf_file.name] = processing_time
    
    return page_image_paths, page_embeddings

@st.cache_data(ttl=3600, show_spinner=False)
def compute_image_embedding(base64_img: str, _cohere_client) -> np.ndarray | None:
    """Computes an embedding for an image using Cohere's Embed-4 model."""
    try:
        start_time = time.time()
        api_response = _cohere_client.embed(
            model="embed-v4.0",
            input_type="search_document",
            embedding_types=["float"],
            images=[base64_img],
        )
        response_time = time.time() - start_time
        track_api_usage("cohere", operation_type="image_embedding", success=True, response_time=response_time)
        
        if api_response.embeddings and api_response.embeddings.float:
            return np.asarray(api_response.embeddings.float[0])
        else:
            track_api_usage("cohere", operation_type="image_embedding", success=False)
            st.warning("Could not get embedding. API response might be empty.")
            return None
    except Exception as e:
        track_api_usage("cohere", operation_type="image_embedding", success=False)
        st.error(f"Error computing embedding: {e}")
        return None

def expand_query(original_query, co_client):
    """Enhanced query expansion with semantic understanding and intent classification."""
    try:
        # First, classify the query intent with French-optimized prompt
        start_time = time.time()
        intent_response = co_client.chat(
            message=get_french_intent_prompt(original_query),
            model="command-r-plus"
        )
        response_time = time.time() - start_time
        track_api_usage("cohere", operation_type="intent_classification", success=True, response_time=response_time)
        intent = intent_response.text.strip().lower()
        
        # Generate context-aware query expansions with French-optimized prompt
        start_time = time.time()
        expansion_prompt = get_french_query_expansion_prompt(original_query, intent)
        
        response = co_client.chat(
            message=expansion_prompt,
            model="command-r-plus"
        )
        response_time = time.time() - start_time
        track_api_usage("cohere", operation_type="query_expansion", success=True, response_time=response_time)
        
        try:
            expanded_queries = json.loads(response.text)
            if isinstance(expanded_queries, list):
                if original_query not in expanded_queries:
                    expanded_queries.insert(0, original_query)
                expanded_queries = expanded_queries[:5]
                return expanded_queries, intent
        except:
            pass
        
        return [original_query], intent
    except Exception as e:
        st.error(f"Error expanding query: {e}")
        return [original_query], "unknown"

def rerank_results(query, candidate_paths, co_client, model_name="rerank-english-v2.0"):
    """Re-rank candidate results using Cohere's reranking capability."""
    image_descriptions = []
    
    for path in candidate_paths:
        try:
            img_data = base64_from_image(path)
            model = genai.GenerativeModel('gemini-pro-vision')
            response = model.generate_content([
                get_french_document_description_prompt(),
                {"mime_type": "image/jpeg", "data": img_data.split(",")[1]}
            ])
            description = response.text
            image_descriptions.append({"path": path, "description": description})
        except Exception as e:
            st.error(f"Error generating description for {path}: {e}")
            image_descriptions.append({"path": path, "description": "Contenu de l'image"})
    
    try:
        rerank_results = co_client.rerank(
            query=query,
            documents=[item["description"] for item in image_descriptions],
            model=model_name,
            top_n=len(candidate_paths)
        )
        
        reranked_paths = []
        for result in rerank_results.results:
            reranked_paths.append(candidate_paths[result.index])
        
        return reranked_paths
    except Exception as e:
        st.error(f"Error reranking: {e}")
        return candidate_paths

def search(question: str, co_client: cohere.Client, embeddings: np.ndarray, image_paths: list[str], top_k: int = 3) -> list[tuple[str, float]] | None:
    """Finds the most relevant image paths for a given question with their similarity scores."""
    if not co_client or embeddings is None or embeddings.size == 0 or not image_paths:
        st.warning("Search prerequisites not met (client, embeddings, or paths missing/empty).")
        return None
    if embeddings.shape[0] != len(image_paths):
         st.error(f"Mismatch between embeddings count ({embeddings.shape[0]}) and image paths count ({len(image_paths)}). Cannot perform search.")
         return None

    try:
        # Compute the embedding for the query
        api_response = co_client.embed(
            model="embed-v4.0",
            input_type="search_query",
            embedding_types=["float"],
            texts=[question],
        )

        if not api_response.embeddings or not api_response.embeddings.float:
            st.error("Failed to get query embedding.")
            return None

        query_emb = np.asarray(api_response.embeddings.float[0])

        # Ensure query embedding has the correct shape for dot product
        if query_emb.shape[0] != embeddings.shape[1]:
            st.error(f"Query embedding dimension ({query_emb.shape[0]}) does not match document embedding dimension ({embeddings.shape[1]}).")
            return None

        # Compute cosine similarities
        cos_sim_scores = np.dot(query_emb, embeddings.T)

        # Get the top-k most relevant images with their scores
        top_indices = np.argsort(cos_sim_scores)[-top_k:][::-1]  # Sort in descending order
        results = [(image_paths[idx], float(cos_sim_scores[idx])) for idx in top_indices]
        
        return results
    except Exception as e:
        st.error(f"Error during search: {e}")
        return None

def collect_search_feedback(query, results, search_time):
    """Collect feedback and metrics for search results"""
    feedback = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "results": [{"path": path, "score": score} for path, score in results],
        "search_time": search_time,
        "feedback": None,
        "user_rating": None
    }
    st.session_state.feedback_data["search_feedback"].append(feedback)
    st.session_state.feedback_data["performance_metrics"]["search_times"].append(search_time)
    
    # Update analytics
    st.session_state.analytics_data["user_engagement"]["total_queries"] += 1
    
    # Save analytics data after updating
    save_analytics_data()
    
    return feedback

def collect_answer_feedback(question, answer, answer_time):
    """Collect feedback and metrics for generated answers"""
    feedback_list = st.session_state.feedback_data["answer_feedback"]
    # Check if an entry for this question/answer already exists
    for entry in feedback_list:
        if entry["question"] == question and entry["answer"] == answer:
            # Update answer_time if needed
            entry["answer_time"] = answer_time
            save_analytics_data()
            return entry
    # If not found, add new
    feedback = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "answer_time": answer_time,
        "feedback": None
    }
    feedback_list.append(feedback)
    st.session_state.feedback_data["performance_metrics"]["answer_times"].append(answer_time)
    save_analytics_data()
    return feedback

def update_analytics(query, intent, success):
    """Update analytics data with query information"""
    # Update query patterns
    if query not in st.session_state.analytics_data["query_patterns"]:
        st.session_state.analytics_data["query_patterns"][query] = 0
    st.session_state.analytics_data["query_patterns"][query] += 1
    
    # Update intent distribution
    if intent not in st.session_state.analytics_data["intent_distribution"]:
        st.session_state.analytics_data["intent_distribution"][intent] = 0
    st.session_state.analytics_data["intent_distribution"][intent] += 1
    
    # Update success rates
    if intent not in st.session_state.analytics_data["success_rates"]:
        st.session_state.analytics_data["success_rates"][intent] = {"success": 0, "total": 0}
    st.session_state.analytics_data["success_rates"][intent]["total"] += 1
    if success:
        st.session_state.analytics_data["success_rates"][intent]["success"] += 1
        st.session_state.analytics_data["user_engagement"]["successful_queries"] += 1
    
    # Save analytics data after updating
    save_analytics_data()

def display_analytics_dashboard():
    """Display a modern, attractive analytics dashboard in the admin panel with comprehensive metrics and modern charts."""
    sns.set_theme(style="darkgrid")
    st.markdown("""
    <style>
    .analytics-header {
        font-size: 2.5rem;
        color: #6366F1;
        margin-bottom: 2.5rem;
        text-align: center;
        font-weight: bold;
        letter-spacing: 1px;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%);
        border-radius: 18px;
        padding: 2rem;
        box-shadow: 0 6px 18px rgba(99,102,241,0.08);
        transition: transform 0.3s;
        margin-bottom: 1.5rem;
    }
    .metric-card:hover {
        transform: translateY(-6px) scale(1.03);
    }
    .metric-value {
        font-size: 2.8rem;
        font-weight: bold;
        color: #6366F1;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 1.2rem;
        color: #64748B;
    }
    .chart-container {
        background: #fff;
        border-radius: 18px;
        padding: 2rem;
        box-shadow: 0 6px 18px rgba(99,102,241,0.08);
        margin: 1.5rem 0;
    }
    .tab-header {
        font-size: 1.7rem;
        color: #6366F1;
        margin-bottom: 2rem;
        font-weight: bold;
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }
    .feedback-section {
        background: #f8fafc;
        border-radius: 18px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 6px 18px rgba(99,102,241,0.08);
    }
    .feedback-header {
        font-size: 1.3rem;
        color: #6366F1;
        margin-bottom: 1rem;
        font-weight: bold;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="analytics-header">📊 System Analytics Dashboard</div>', unsafe_allow_html=True)

    analytics_tabs = st.tabs([
        "📈 Overview", "🔍 Query Analysis", "⚡ Performance Metrics", "💬 User Feedback", "🕒 Recent Feedback"
    ])

    # --- Overview Tab ---
    with analytics_tabs[0]:
        st.markdown('<div class="tab-header">📈 System Overview</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{st.session_state.analytics_data['user_engagement']['total_queries']}</div>
                <div class="metric-label">Total Queries</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            success_rate = (
                st.session_state.analytics_data["user_engagement"]["successful_queries"] /
                max(1, st.session_state.analytics_data["user_engagement"]["total_queries"]) * 100
            )
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{success_rate:.1f}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{st.session_state.analytics_data['user_engagement']['feedback_count']}</div>
                <div class="metric-label">Feedback Count</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            avg_response = 0
            answer_times = st.session_state.feedback_data["performance_metrics"].get("answer_times", [])
            if answer_times:
                avg_response = sum(answer_times) / len(answer_times)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_response:.2f}s</div>
                <div class="metric-label">Avg. Response Time</div>
            </div>
            """, unsafe_allow_html=True)
        # Intent distribution
        if st.session_state.analytics_data["intent_distribution"]:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            intents = list(st.session_state.analytics_data["intent_distribution"].keys())
            counts = list(st.session_state.analytics_data["intent_distribution"].values())
            colors = sns.color_palette("cool", len(intents))
            wedges, texts, autotexts = ax.pie(counts, labels=intents, colors=colors,
                   autopct='%1.1f%%', shadow=False, startangle=90, pctdistance=0.85)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(12)
            centre_circle = plt.Circle((0,0),0.70,fc='white')
            fig.gca().add_artist(centre_circle)
            ax.axis('equal')
            plt.title('Query Intent Distribution', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        # --- Delete Analytics Data Button ---
        if st.button("Delete All Analytics Data", type="primary"):
            st.session_state.analytics_data = {
                "query_patterns": {},
                "intent_distribution": {},
                "success_rates": {},
                "user_engagement": {
                    "total_queries": 0,
                    "successful_queries": 0,
                    "feedback_count": 0
                }
            }
            st.session_state.feedback_data = {
                "search_feedback": [],
                "answer_feedback": [],
                "performance_metrics": {
                    "search_times": [],
                    "answer_times": [],
                    "query_success": []
                }
            }
            if os.path.exists(ANALYTICS_FILE):
                os.remove(ANALYTICS_FILE)
            st.session_state.analytics_reset = True
            st.success("Analytics data deleted. The dashboard will reset after the next action.")
            st.rerun()

    # --- Query Analysis Tab ---
    with analytics_tabs[1]:
        st.markdown('<div class="tab-header">🔍 Query Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Most Common Queries")
        query_data = pd.DataFrame([
            {"Query": q, "Count": c}
            for q, c in st.session_state.analytics_data["query_patterns"].items()
        ])
        if not query_data.empty:
            query_data = query_data.sort_values("Count", ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            top_queries = query_data.head(10)
            bars = ax.barh(top_queries["Query"], top_queries["Count"], color=sns.color_palette("crest", len(top_queries)))
            ax.set_xlabel('Count', fontsize=12)
            ax.set_title('Top 10 Most Common Queries', fontsize=14, fontweight='bold')
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, f'{int(width)}', ha='left', va='center', fontsize=11, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No query data available yet.")
        st.markdown('</div>', unsafe_allow_html=True)
        # Success rates by intent
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Success Rates by Intent")
        success_data = []
        for intent, data in st.session_state.analytics_data["success_rates"].items():
            success_rate = data["success"] / max(1, data["total"]) * 100
            success_data.append({
                "Intent": intent,
                "Success Rate": success_rate,
                "Total Queries": data["total"]
            })
        if success_data:
            success_df = pd.DataFrame(success_data)
            fig, ax = plt.subplots(figsize=(10, 6))
            x = range(len(success_df))
            width = 0.35
            bars1 = ax.bar([i - width/2 for i in x], success_df["Success Rate"], width, label='Success Rate (%)', color="#6366F1")
            bars2 = ax.bar([i + width/2 for i in x], success_df["Total Queries"], width, label='Total Queries', color="#A5B4FC")
            ax.set_ylabel('Count/Percentage', fontsize=12)
            ax.set_title('Success Rates by Intent', fontsize=14, fontweight='bold')
            ax.set_xticks(list(x))
            ax.set_xticklabels(success_df["Intent"], rotation=45, ha='right')
            ax.legend()
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No success rate data available yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Performance Metrics Tab ---
    with analytics_tabs[2]:
        st.markdown('<div class="tab-header">⚡ Performance Metrics</div>', unsafe_allow_html=True)
        # Search time distribution
        if st.session_state.feedback_data["performance_metrics"]["search_times"]:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("Search Time Distribution")
            search_times = st.session_state.feedback_data["performance_metrics"]["search_times"]
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(search_times, bins=15, kde=True, color="#6366F1", ax=ax)
            ax.set_title('Search Time Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        # Answer time distribution
        if st.session_state.feedback_data["performance_metrics"]["answer_times"]:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("Answer Time Distribution")
            answer_times = st.session_state.feedback_data["performance_metrics"]["answer_times"]
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(answer_times, bins=15, kde=True, color="#A5B4FC", ax=ax)
            ax.set_title('Answer Time Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

    # --- User Feedback Tab ---
    with analytics_tabs[3]:
        st.markdown('<div class="tab-header">💬 User Feedback</div>', unsafe_allow_html=True)
        # Search feedback
        if st.session_state.feedback_data["search_feedback"]:
            st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
            st.markdown('<div class="feedback-header">📊 Detailed Search Feedback</div>', unsafe_allow_html=True)
            search_feedback = pd.DataFrame(st.session_state.feedback_data["search_feedback"])
            if not search_feedback.empty:
                st.dataframe(search_feedback, use_container_width=True, hide_index=True)
            else:
                st.info("No search feedback available yet.")
            st.markdown('</div>', unsafe_allow_html=True)
        # Answer feedback
        if st.session_state.feedback_data["answer_feedback"]:
            st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
            st.markdown('<div class="feedback-header">💬 Detailed Answer Feedback</div>', unsafe_allow_html=True)
            answer_feedback = pd.DataFrame(st.session_state.feedback_data["answer_feedback"])
            # Remove user_rating column if it exists
            if 'user_rating' in answer_feedback.columns:
                answer_feedback = answer_feedback.drop(columns=['user_rating'])
            if not answer_feedback.empty:
                st.dataframe(answer_feedback, use_container_width=True, hide_index=True)
            else:
                st.info("No answer feedback available yet.")
            st.markdown('</div>', unsafe_allow_html=True)

    # --- Recent Feedback Tab ---
    with analytics_tabs[4]:
        st.markdown('<div class="tab-header">🕒 Recent Feedback</div>', unsafe_allow_html=True)
        # Only show Answer feedback
        feedback_entries = []
        for entry in st.session_state.feedback_data["answer_feedback"]:
            feedback_entries.append(entry)
        if feedback_entries:
            feedback_entries = sorted(feedback_entries, key=lambda x: x.get("timestamp", ""), reverse=True)
            df_recent = pd.DataFrame(feedback_entries[:10])
            # Only show selected columns
            show_cols = [col for col in ["timestamp", "question", "answer", "answer_time", "feedback"] if col in df_recent.columns]
            st.dataframe(df_recent[show_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No recent feedback available yet.")

def enhanced_search(query, co_client, doc_embeddings, image_paths, top_k=5):
    """
    Enhanced search with weighted scoring, semantic clustering, and improved result aggregation.
    """
    start_time = time.time()
    
    if not co_client or doc_embeddings is None or len(image_paths) == 0:
        return []
    
    try:
        # Step 1: Expand the query and get intent
        expanded_queries, intent = expand_query(query, co_client)
        
        # Step 2: Perform search with expanded queries and weighted scoring
        all_results = []
        query_weights = {
            "Factual": 1.2,  # Higher weight for factual queries
            "Analytical": 1.1,  # Slightly higher weight for analytical queries
            "Summarization": 1.0,  # Base weight for summarization
            "Verification": 1.3,  # Highest weight for verification queries
            "unknown": 1.0  # Default weight
        }
        
        # Get base weight for the query intent
        base_weight = query_weights.get(intent, 1.0)
        
        for expanded_query in expanded_queries:
            # Generate embedding for the expanded query
            track_api_usage("cohere")  # Track Cohere API usage
            response = co_client.embed(
                model="embed-v4.0",
                input_type="search_query",
                embedding_types=["float"],
                texts=[expanded_query],
            )
            
            if not response.embeddings or not response.embeddings.float:
                continue
                
            query_embedding = np.asarray(response.embeddings.float[0])
            similarities = np.dot(query_embedding, doc_embeddings.T)
            
            # Apply weights based on query type and position
            for i, score in enumerate(similarities):
                # Weight adjustment based on query position (original query gets higher weight)
                position_weight = 1.2 if expanded_query == query else 1.0
                
                # Calculate final weighted score
                weighted_score = float(score) * base_weight * position_weight
                
                all_results.append({
                    "path": image_paths[i],
                    "score": weighted_score,
                    "query": expanded_query,
                    "intent": intent,
                    "base_score": float(score)
                })
        
        # Step 3: Semantic clustering of results
        if len(all_results) > 1:
            # Convert results to numpy array for clustering
            result_scores = np.array([r["base_score"] for r in all_results])
            
            # Simple clustering using score distribution
            score_mean = np.mean(result_scores)
            score_std = np.std(result_scores)
            
            # Group results into clusters based on score distribution
            clusters = {}
            for result in all_results:
                score = result["base_score"]
                if score > score_mean + score_std:
                    cluster = "high"
                elif score > score_mean:
                    cluster = "medium"
                else:
                    cluster = "low"
                
                if cluster not in clusters:
                    clusters[cluster] = []
                clusters[cluster].append(result)
        
        # Step 4: Smart result aggregation
        unique_results = {}
        for result in all_results:
            path = result["path"]
            if path not in unique_results:
                unique_results[path] = result
            else:
                # Update score if new result has higher score
                if result["score"] > unique_results[path]["score"]:
                    unique_results[path] = result
                # If scores are close, prefer results from original query
                elif abs(result["score"] - unique_results[path]["score"]) < 0.1:
                    if result["query"] == query:
                        unique_results[path] = result
        
        # Step 5: Diversified result selection
        final_results = []
        # First, add high-scoring results
        high_scoring = [r for r in unique_results.values() if r["base_score"] > score_mean + score_std]
        final_results.extend(high_scoring)
        
        # Then, add medium-scoring results if we need more
        if len(final_results) < top_k:
            medium_scoring = [r for r in unique_results.values() 
                            if score_mean <= r["base_score"] <= score_mean + score_std]
            final_results.extend(medium_scoring)
        
        # Finally, add low-scoring results if we still need more
        if len(final_results) < top_k:
            low_scoring = [r for r in unique_results.values() 
                         if r["base_score"] < score_mean]
            final_results.extend(low_scoring)
        
        # Sort by score and get top results
        final_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Collect feedback and metrics
        search_time = time.time() - start_time
        feedback = collect_search_feedback(query, [(r["path"], r["score"]) for r in final_results[:top_k]], search_time)
        
        # Update analytics
        update_analytics(query, intent, len(final_results) > 0)
        
        # Return only the path and score pairs
        return [(r["path"], r["score"]) for r in final_results[:top_k]]
    
    except Exception as e:
        st.error(f"Error during enhanced search: {e}")
        return search(query, co_client, doc_embeddings, image_paths, top_k)

def generate_system_message(question_history, img_paths, context_info=None):
    """Generates an advanced system message for the chatbot based on conversation history and available context."""
    document_types = []
    for path in img_paths:
        if "page_" in path:
            pdf_name = os.path.basename(os.path.dirname(path))
            page_num = os.path.basename(path).replace("page_", "").replace(".png", "")
            document_types.append(f"PDF '{pdf_name}' (page {page_num})")
        else:
            document_types.append(f"Image '{os.path.basename(path)}'")
    
    conversation_context = ""
    if question_history and len(question_history) > 1:
        recent_questions = [msg["content"] for msg in question_history[-3:] if msg["role"] == "user"]
        conversation_context = f"\n\nContexte de conversation récent:\n" + "\n".join([f"- {q}" for q in recent_questions])
    
    # Use French-optimized system message
    return get_french_system_message(
        document_types=", ".join(document_types),
        conversation_context=conversation_context,
        additional_context=context_info or ""
    )

# Update the answer function to use the system message
def answer(question: str, img_paths: list[str], stream=True, conversation_history=None) -> str:
    """
    Enhanced answer generation with better context management and verification.
    """
    start_time = time.time()
    
    if not img_paths:
        return "Answering prerequisites not met (No image paths provided)."
    
    # Check if all image paths exist
    missing_paths = [path for path in img_paths if not os.path.exists(path)]
    if missing_paths:
        return f"Answering prerequisites not met (Image files missing at: {', '.join(missing_paths)})."
    
    try:
        # Generate system message based on conversation context and available images
        system_message = generate_system_message(
            question_history=conversation_history or [], 
            img_paths=img_paths
        )
        
        # Enhanced comprehensive prompt with chain-of-thought and context-aware instructions
        prompt_parts = [
            system_message,
            f"Question: {question}"
        ]
        
        # Add all images to the prompt
        for img_path in img_paths:
            img = PIL.Image.open(img_path)
            prompt_parts.append(img)

        # Instantiate the model
        model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
        
        if stream:
            response_start = time.time()
            response = model.generate_content(contents=prompt_parts, stream=True)
            response_time = time.time() - response_start
            track_api_usage("google", operation_type="streaming_answer", success=True, response_time=response_time)
            return response
        else:
            response_start = time.time()
            response = model.generate_content(contents=prompt_parts)
            response_time = time.time() - response_start
            track_api_usage("google", operation_type="answer_generation", success=True, response_time=response_time)
            
            if hasattr(response, 'text'):
                llm_answer = response.text
                # Collect feedback and metrics
                answer_time = time.time() - start_time
                feedback = collect_answer_feedback(question, llm_answer, answer_time)
                return llm_answer
            elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                reason = response.prompt_feedback.block_reason
                track_api_usage("google", operation_type="answer_generation", success=False)
                st.error(f"Content generation blocked: {reason}")
                return f"Failed to generate answer due to content blocking: {reason}"
            else:
                track_api_usage("google", operation_type="answer_generation", success=False)
                st.error("Received an unexpected response format from Gemini.")
                return "Failed to generate answer due to unexpected response format."

    except Exception as e:
        track_api_usage("google", operation_type="answer_generation", success=False)
        st.error(f"Error during answer generation: {e}")
        if "API key not valid" in str(e):
            return "Failed to generate answer: Invalid Google API Key."
        return f"Failed to generate answer: {e}"

def toggle_references(message_id):
    """Toggle the visibility of references for a specific message."""
    if message_id in st.session_state.show_references:
        st.session_state.show_references[message_id] = not st.session_state.show_references[message_id]
    else:
        st.session_state.show_references[message_id] = True

def save_feedback(index, answer_content=None):
    st.session_state.history[index]["feedback"] = st.session_state[f"feedback_{index}"]
    st.session_state[f"feedback_answer_{index}"] = answer_content

def display_chat_history():
    """Display the chat history with feedback functionality."""
    if st.session_state.doc_embeddings is not None and st.session_state.image_paths:
        if not st.session_state.history:
            return
        for i, message in enumerate(st.session_state.history):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    feedback = message.get("feedback", None)
                    st.session_state[f"feedback_{i}"] = feedback
                    # Create toolbar container
                    st.markdown('<div class="chat-toolbar">', unsafe_allow_html=True)
                    # Sources button
                    if "references" in message:
                        if st.button("Sources", key=f"ref_toggle_{i}", use_container_width=False):
                            toggle_references(f"msg_{i}")
                    # Set the answer in session state before rendering feedback button
                    st.session_state[f"feedback_answer_{i}"] = message["content"]
                    # Feedback buttons - directly call handle_feedback
                    st.feedback(
                        "thumbs",
                        key=f"feedback_{i}",
                        on_change=handle_feedback,
                        args=[i],
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    # Show references if toggled
                    if st.session_state.show_references.get(f"msg_{i}", False):
                        st.write("#### References:")
                        cols = st.columns(min(3, len(message["references"])))
                        for j, ((img_path, score), col) in enumerate(zip(message["references"], cols)):
                            with col:
                                st.image(img_path, caption=f"Relevance: {score:.2f}", use_container_width=True)
                                filename = os.path.basename(img_path)
                                if "page_" in filename:
                                    pdf_folder = os.path.basename(os.path.dirname(img_path))
                                    page_num = filename.replace("page_", "").replace(".png", "")
                                    st.caption(f"{pdf_folder} - Page {page_num}")
                                else:
                                    st.caption(filename)
                elif "image" in message and message["image"]:
                    st.image(message["image"], use_container_width=True)

def handle_feedback(message_index):
    """Handle feedback submission"""
    feedback_data = st.session_state.get(f"feedback_{message_index}")
    answer = st.session_state.get(f"feedback_answer_{message_index}")

    # If feedback_data is an int, map it to thumbs up/down
    if isinstance(feedback_data, int):
        if feedback_data == 1:
            feedback_icon = "👍"
        elif feedback_data == 0:
            feedback_icon = "👎"
        else:
            feedback_icon = "No feedback given"
        feedback_text = feedback_icon
    elif isinstance(feedback_data, dict):
        feedback_icon = feedback_data.get("score", "")
        feedback_text = feedback_icon if feedback_icon in ["👍", "👎"] else "No feedback given"
    else:
        return

    question = None
    if message_index < len(st.session_state.history):
        # Find the previous user message
        for idx in range(message_index - 1, -1, -1):
            if st.session_state.history[idx]["role"] == "user":
                question = st.session_state.history[idx]["content"]
                break
    # Ensure answer is a readable string
    if isinstance(answer, (dict, list)):
        try:
            answer = json.dumps(answer, ensure_ascii=False)
        except Exception:
            answer = str(answer)
    elif not isinstance(answer, str):
        answer = str(answer)
    answer_time = None
    if message_index < len(st.session_state.history):
        msg = st.session_state.history[message_index]
        if "answer_time" in msg:
            answer_time = msg["answer_time"]
        else:
            answer_time = None
    # Update the existing feedback entry if it exists
    feedback_list = st.session_state.feedback_data["answer_feedback"]
    updated = False
    for entry in feedback_list:
        if entry["question"] == question and entry["answer"] == answer:
            entry["feedback"] = feedback_text
            entry["timestamp"] = datetime.now().isoformat()
            updated = True
            break
    if not updated:
        # If not found, add new (shouldn't happen if collect_answer_feedback is used properly)
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "answer_time": answer_time,
            "feedback": feedback_text
        }
        feedback_list.append(feedback_entry)
    # Always increment feedback count when user gives feedback
    st.session_state.analytics_data["user_engagement"]["feedback_count"] += 1
    # Save analytics data after updating
    save_analytics_data()
    message_placeholder = st.empty()
    if feedback_icon == "👍":
        message_placeholder.success("Thank you for your positive feedback! 👍")
    elif feedback_icon == "👎":
        message_placeholder.warning("Thank you for your feedback. We'll use it to improve! 👎")
    else:
        message_placeholder.info("Feedback received.")
    time.sleep(3)
    message_placeholder.empty()
    if message_index < len(st.session_state.history):
        st.session_state.history[message_index]["feedback"] = feedback_data
        save_chat_history(st.session_state.history)

def clear_chat():
    """Clear the chat history and save empty state."""
    st.session_state.history = []
    save_chat_history([])
    st.rerun()

# Add a callback function for the text input
def submit_question():
    """Handle form submission and process the question."""
    if st.session_state.question:
        question = st.session_state.question
        process_question(question)
        # Note: We don't clear the input here as it would cause the error

def on_send_message():
    """Handle sending a message."""
    if st.session_state.question:
        # Process the question without trying to clear the input field
        process_question(st.session_state.question)
        # The input field will be cleared on the next rerun

def process_question(user_input):
    question = user_input
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    try:
        if co and st.session_state.doc_embeddings is not None and st.session_state.image_paths:
            search_results = enhanced_search(query=question, co_client=co, doc_embeddings=st.session_state.doc_embeddings, image_paths=st.session_state.image_paths, top_k=3)
            if search_results:
                relevant_paths = [path for path, _ in search_results]
                with st.chat_message("assistant"):
                    content_col, _ = st.columns([0.9, 0.1])
                    full_response = ""
                    start_time = time.time()
                    with content_col:
                        response_placeholder = st.empty()
                        # Pass conversation history for context-aware responses
                        response_stream = answer(
                            question=question, 
                            img_paths=relevant_paths, 
                            stream=True,
                            conversation_history=st.session_state.history
                        )
                        try:
                            for chunk in response_stream:
                                if hasattr(chunk, 'text'):
                                    full_response += chunk.text
                                    response_placeholder.markdown(full_response + "▌")
                                elif isinstance(chunk, str):
                                    full_response += chunk
                                    response_placeholder.markdown(full_response + "▌")
                            response_placeholder.markdown(full_response)
                        except Exception as e:
                            st.error(f"Error during response streaming: {str(e)}")
                            try:
                                fallback_response = answer(
                                    question=question, 
                                    img_paths=relevant_paths, 
                                    stream=False,
                                    conversation_history=st.session_state.history
                                )
                                if isinstance(fallback_response, str):
                                    full_response = fallback_response
                                    response_placeholder.markdown(full_response)
                                else:
                                    st.error("Failed to generate response")
                                    return
                            except Exception as fallback_error:
                                st.error(f"Fallback response generation failed: {str(fallback_error)}")
                                return
                    # After streaming is done, collect feedback with the full answer
                    answer_time = time.time() - start_time
                    collect_answer_feedback(question, full_response, answer_time)
                st.session_state.history.append({"role": "assistant", "content": full_response, "references": search_results, "answer_time": answer_time})
                save_chat_history(st.session_state.history)
            else:
                st.error("No relevant documents found in the loaded data.")
        else:
            if not co:
                st.error("API client not initialized. Please check your API keys.")
            elif st.session_state.doc_embeddings is None or not st.session_state.image_paths:
                st.error("No documents loaded. Please upload documents first.")
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
    st.session_state.question = ""
    st.rerun()

# --- Main Chat UI ---
if st.session_state.show_admin:
    st.markdown('<div class="admin-header">🔐 Admin Panel</div>', unsafe_allow_html=True)
    
    # Login screen if not authenticated
    if not st.session_state.admin_authenticated:
        with st.form("admin_login_form"):
            st.markdown('<div class="admin-subheader">Admin Login</div>', unsafe_allow_html=True)
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if verify_credentials(username, password):
                    st.session_state.admin_authenticated = True
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    else:
        # Admin navigation
        admin_tabs = st.tabs(["API Keys", "Document Statistics", "Settings"])
        
        # API Keys tab
        with admin_tabs[0]:
            st.markdown('<div class="admin-subheader">API Key Management</div>', unsafe_allow_html=True)
            
            # Get current keys
            keys = get_api_keys()
            
            with st.form("admin_api_keys_form"):
                st.markdown('<div class="admin-subheader">Update API Keys</div>', unsafe_allow_html=True)
                
                admin_cohere_key = st.text_input("Cohere API Key", value=keys["cohere_key"])
                admin_google_key = st.text_input("Google API Key", value=keys["google_key"])
                
                submit = st.form_submit_button("Save Keys")
                
                if submit:
                    if update_api_keys(admin_cohere_key, admin_google_key):
                        st.success("API keys updated successfully! Please refresh the page to apply changes.")
                    else:
                        st.error("Failed to update API keys")
        
        # Document Statistics tab
        with admin_tabs[1]:
            display_document_statistics()
        
        # Settings tab
        with admin_tabs[2]:
            st.markdown('<div class="admin-subheader">Admin Settings</div>', unsafe_allow_html=True)
            
            # Password change section
            st.markdown('<div class="admin-subheader">Change Password</div>', unsafe_allow_html=True)
            
            with st.form("admin_password_change_form"):
                current_password = st.text_input("Current Password", type="password")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm New Password", type="password")
                
                submit = st.form_submit_button("Change Password")
                
                if submit:
                    if not verify_credentials("admin", current_password):
                        st.error("Current password is incorrect")
                    elif new_password != confirm_password:
                        st.error("New passwords do not match")
                    elif len(new_password) < 6:
                        st.error("New password must be at least 6 characters long")
                    else:
                        if change_password(new_password):
                            st.success("Password changed successfully!")
                        else:
                            st.error("Failed to change password")
            
            # Logout button
            if st.button("Logout"):
                st.session_state.admin_authenticated = False
                st.rerun()
else:
    # Main chat interface
    main_content = st.container()
    with main_content:
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        display_chat_history()
        st.markdown('</div>', unsafe_allow_html=True)

    # Only show chat input if API client is initialized
    if co:
        if st.session_state.doc_embeddings is None or not st.session_state.image_paths:
            st.info("""
            📚 **No documents loaded yet!** To start asking questions, you need to either:
            1. Click the "Load Previous Data" button in the sidebar to load previously processed documents
            2. Upload new documents using the chat input below
            Once you've loaded or uploaded documents, you can start asking questions about them.
            """)
    
    # Modified chat input to accept files
    prompt = st.chat_input(
        "Ask about your documents or upload files...",
        key="question_input",
        accept_file=True,
        file_type=["png", "jpg", "jpeg", "pdf"]
    )
    
    if prompt:
        # Handle file uploads if any
        if prompt.files:
            uploaded_files = prompt.files
            uploaded_file_names = sorted([f.name for f in uploaded_files])
            # Always process files immediately after upload
            st.session_state.last_uploaded_names = uploaded_file_names
            st.write(f"Processing {len(uploaded_files)} uploaded file(s)...")
            progress_bar = st.progress(0)
            
            # Create temporary directories
            upload_folder = "uploaded_img"
            os.makedirs(upload_folder, exist_ok=True)
            pdf_page_folder = "pdf_pages"
            os.makedirs(pdf_page_folder, exist_ok=True)
            
            newly_uploaded_paths = []
            newly_uploaded_embeddings = []

            for i, uploaded_file in enumerate(uploaded_files):
                base_name = uploaded_file.name
                is_pdf = uploaded_file.type == "application/pdf"
                already_processed = False
                if is_pdf:
                    pdf_page_prefix = os.path.join(pdf_page_folder, os.path.splitext(base_name)[0], "page_")
                    if any(p.startswith(pdf_page_prefix) for p in st.session_state.image_paths):
                        already_processed = True
                else:
                    img_path = os.path.join(upload_folder, base_name)
                    if img_path in st.session_state.image_paths:
                        already_processed = True

                if not already_processed:
                    try:
                        if is_pdf:
                            uploaded_file.seek(0)  # Ensure pointer is at start
                            with st.spinner(f"Processing PDF: {base_name}"):
                                pdf_page_paths, pdf_page_embeddings = process_pdf_file(uploaded_file, cohere_client=co, base_output_folder=pdf_page_folder)
                                if pdf_page_paths and pdf_page_embeddings:
                                    newly_uploaded_paths.extend(pdf_page_paths)
                                    newly_uploaded_embeddings.extend(pdf_page_embeddings)
                        else: # Process as image
                            img_path = os.path.join(upload_folder, uploaded_file.name)
                            with open(img_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            base64_img = base64_from_image(img_path)
                            emb = compute_image_embedding(base64_img, _cohere_client=co)
                            if emb is not None:
                                newly_uploaded_paths.append(img_path)
                                newly_uploaded_embeddings.append(emb)
                            else:
                                st.warning(f"Could not embed uploaded image: {uploaded_file.name}. Skipping.")
                                if os.path.exists(img_path):
                                    os.remove(img_path)
                    except Exception as e:
                        st.error(f"Error processing file {uploaded_file.name}: {e}")
                # Update progress bar after each file
                progress_bar.progress((i + 1) / len(uploaded_files))

            progress_bar.empty() # Remove progress bar

            # Update session state if new embeddings were generated
            if newly_uploaded_paths and newly_uploaded_embeddings:
                st.session_state.image_paths.extend(newly_uploaded_paths)
                
                new_embeddings_array = np.array(newly_uploaded_embeddings)

                if st.session_state.doc_embeddings is None or st.session_state.doc_embeddings.size == 0:
                    st.session_state.doc_embeddings = new_embeddings_array
                else:
                    # Ensure dimensions match before vstack
                    if st.session_state.doc_embeddings.shape[1] == new_embeddings_array.shape[1]:
                        st.session_state.doc_embeddings = np.vstack((st.session_state.doc_embeddings, new_embeddings_array))
                    else:
                        st.error("Dimension mismatch between existing and new embeddings. Cannot combine.")
                
                st.success(f"Processed {len(newly_uploaded_paths)} new items.")
                
                # Save the updated data for future sessions
                save_persistent_data(st.session_state.doc_embeddings, st.session_state.image_paths)
                
                # Rerun to update the main UI state based on new embeddings
                st.rerun()
        
        # Handle text input if any
        if prompt.text and prompt.text.strip():
            st.session_state.question = prompt.text.strip()
            process_question(prompt.text.strip())

if st.session_state.get("analytics_reset", False):
    st.session_state.analytics_reset = False
    st.rerun()

# Workspace management panel (like admin panel)
elif st.session_state.get('show_workspace', False):
    st.markdown('<div class="workspace-header">🗂️ Workspace Management</div>', unsafe_allow_html=True)
    workspaces = load_workspaces()
    active_ws = get_active_workspace()
    workspace_tabs = st.tabs(["Workspaces", "Create Workspace", "Upload Data"])
    with workspace_tabs[0]:
        st.markdown("### Existing Workspaces")
        for ws_name in workspaces:
            is_active = (ws_name == active_ws)
            col1, col2 = st.columns([0.7, 0.3])
            with col1:
                st.markdown(f"- {'**'+ws_name+'**' if is_active else ws_name}")
            with col2:
                if not is_active:
                    if st.button(f"Switch to {ws_name}", key=f"switch_{ws_name}"):
                        set_active_workspace(ws_name)
                        st.success(f"Switched to workspace: {ws_name}")
                        st.rerun()
    with workspace_tabs[1]:
        st.markdown("### Create New Workspace")
        with st.form("create_workspace_form"):
            new_ws_name = st.text_input("Workspace Name")
            submit_ws = st.form_submit_button("Create Workspace")
            if submit_ws and new_ws_name:
                ok, msg = create_workspace(new_ws_name)
                if ok:
                    st.success(msg)
                    set_active_workspace(new_ws_name)
                    st.rerun()
                else:
                    st.error(msg)
    with workspace_tabs[2]:
        st.markdown(f"### Upload Data for Workspace: {active_ws}")
        upload_files = st.file_uploader(
            "Upload PDFs or images for this workspace",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="workspace_file_uploader"
        )
        if upload_files:
            if st.button("Upload and Process Files", key="workspace_upload_btn"):
                st.write(f"Processing {len(upload_files)} uploaded file(s) for workspace {active_ws}...")
                progress_bar = st.progress(0)
                # Prepare workspace-specific paths
                workspaces = load_workspaces()
                ws = workspaces.get(active_ws)
                upload_folder = "uploaded_img"
                os.makedirs(upload_folder, exist_ok=True)
                pdf_page_folder = "pdf_pages"
                os.makedirs(pdf_page_folder, exist_ok=True)
                # Load current workspace data
                image_paths = []
                doc_embeddings = None
                if ws and os.path.exists(ws["paths_file"]):
                    with open(ws["paths_file"], 'r') as f:
                        image_paths = json.load(f)
                if ws and os.path.exists(ws["embeddings_file"]):
                    doc_embeddings = np.load(ws["embeddings_file"])
                newly_uploaded_paths = []
                newly_uploaded_embeddings = []
                for i, uploaded_file in enumerate(upload_files):
                    base_name = uploaded_file.name
                    is_pdf = uploaded_file.type == "application/pdf"
                    already_processed = False
                    if is_pdf:
                        pdf_page_prefix = os.path.join(pdf_page_folder, os.path.splitext(base_name)[0], "page_")
                        if any(p.startswith(pdf_page_prefix) for p in image_paths):
                            already_processed = True
                    else:
                        img_path = os.path.join(upload_folder, base_name)
                        if img_path in image_paths:
                            already_processed = True
                    if not already_processed:
                        try:
                            if is_pdf:
                                uploaded_file.seek(0)
                                with st.spinner(f"Processing PDF: {base_name}"):
                                    pdf_page_paths, pdf_page_embeddings = process_pdf_file(uploaded_file, cohere_client=co, base_output_folder=pdf_page_folder)
                                if pdf_page_paths and pdf_page_embeddings:
                                    newly_uploaded_paths.extend(pdf_page_paths)
                                    newly_uploaded_embeddings.extend(pdf_page_embeddings)
                            else:
                                img_path = os.path.join(upload_folder, uploaded_file.name)
                                with open(img_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                base64_img = base64_from_image(img_path)
                                emb = compute_image_embedding(base64_img, _cohere_client=co)
                                if emb is not None:
                                    newly_uploaded_paths.append(img_path)
                                    newly_uploaded_embeddings.append(emb)
                                else:
                                    st.warning(f"Could not embed uploaded image: {uploaded_file.name}. Skipping.")
                                    if os.path.exists(img_path):
                                        os.remove(img_path)
                        except Exception as e:
                            st.error(f"Error processing file {uploaded_file.name}: {e}")
                    progress_bar.progress((i + 1) / len(upload_files))
                progress_bar.empty()
                # Update workspace data
                if newly_uploaded_paths and newly_uploaded_embeddings:
                    image_paths.extend(newly_uploaded_paths)
                    new_embeddings_array = np.array(newly_uploaded_embeddings)
                    if doc_embeddings is None or doc_embeddings.size == 0:
                        doc_embeddings = new_embeddings_array
                    else:
                        if doc_embeddings.shape[1] == new_embeddings_array.shape[1]:
                            doc_embeddings = np.vstack((doc_embeddings, new_embeddings_array))
                        else:
                            st.error("Dimension mismatch between existing and new embeddings. Cannot combine.")
                    # Save to workspace files
                    np.save(ws["embeddings_file"], doc_embeddings)
                    with open(ws["paths_file"], 'w') as f:
                        json.dump(image_paths, f)
                    st.success(f"Processed {len(newly_uploaded_paths)} new items for workspace {active_ws}.")
                    st.rerun()
                else:
                    st.warning("No new files were processed. Please check your uploads or if they were already added.")
    if st.button("Close Workspace Panel", key="close_workspace_panel"):
        st.session_state.show_workspace = False
    st.rerun()


# At the end of the script:
if st.session_state.get('should_rerun', False):
    st.session_state.should_rerun = False
    st.rerun()