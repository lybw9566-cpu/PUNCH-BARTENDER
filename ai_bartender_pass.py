import streamlit as st
import pandas as pd
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 0. 门卫系统 (核心修改部分) ---
def check_access():
    """
    门卫函数：如果没有通过验证，就显示登录框并停止运行后面的代码
    """
    # 初始化验证状态
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # 如果已经登录成功，直接放行
    if st.session_state.authenticated:
        return True

    # 如果没登录，显示登录界面
    st.title("🔒 这是一个私密应用")
    st.write("请在下方输入邀请码以继续访问。")
    
    # 获取用户输入
    user_input = st.text_input("请输入邀请码 (Access Key)", type="password")
    
    if st.button("解锁进入"):
        # 检查邀请码是否在我们的“白名单”里
        # 注意：我们会把白名单放在 st.secrets 里
        valid_keys = st.secrets.get("access_keys", [])
        
        if user_input in valid_keys:
            st.session_state.authenticated = True
            st.success("✅ 验证成功！正在加载...")
            st.rerun() # 刷新页面进入主程序
        else:
            st.error("❌ 邀请码无效或已失效")
    
    # 如果没通过验证，返回 False，阻止后续代码运行
    return False

# 执行门卫检查
if not check_access():
    st.stop() # 🛑 停止运行下面的所有代码

# ===========================================
#  以下是原本的 AI 侍酒师代码 (只有通过上面检查才会运行到这里)
# ===========================================

# --- 1. 配置加载 ---
# 优先从 Streamlit Cloud 的 Secrets 读取，如果没有则读取本地 .env
if "OPENAI_API_KEY" in st.secrets:
    API_KEY = st.secrets["OPENAI_API_KEY"]
    BASE_URL = st.secrets["OPENAI_BASE_URL"]
    MODEL_NAME = st.secrets["OPENAI_MODEL_NAME"]
else:
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = os.getenv("OPENAI_BASE_URL")
    MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")

if not API_KEY:
    st.error("❌ 未配置 API Key")
    st.stop()

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
DATA_FILE = "punch_recipes.jsonl"

# ... (保持原本的 Page Config) ...
# 注意：set_page_config 必须是 Streamlit 命令的第一行，
# 但为了配合门卫逻辑，我们需要把它移到最最上面，或者接受这里的小警告。
# 为了代码规范，建议把 st.set_page_config 移到代码文件的第一行（import 之后）。
# 这里为了演示方便，先不移动，Streamlit 可能会报个无害的 Warning。

# --- 2. 数据加载与向量化 (保持不变) ---
@st.cache_resource
def load_data_and_vectors():
    data = []
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        return None, None, None

    df = pd.DataFrame(data)
    df['combined_text'] = (
        df['title'].fillna('') + " " + 
        df['intro_philosophy'].fillna('') + " " + 
        df['ingredients'].astype(str) + " " + 
        df['tags'].astype(str)
    )
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    return df, vectorizer, tfidf_matrix

df, vectorizer, tfidf_matrix = load_data_and_vectors()

if df is None:
    st.error(f"❌ 找不到数据文件 {DATA_FILE}")
    st.stop()

# --- 3. 核心逻辑 (GPT 通用版) ---
def get_ai_recommendation(user_query):
    # === A. 检索 ===
    try:
        user_vec = vectorizer.transform([user_query])
        similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-15:][::-1]
        candidates = df.iloc[top_indices]
    except Exception as e:
        return f"检索系统出错了: {e}", pd.DataFrame()

    # === B. 增强 ===
    context_text = ""
    for idx, row in candidates.iterrows():
        context_text += f"""
        [酒名: {row['title']}]
        [原料: {row['ingredients']}]
        [步骤: {row['instructions']}]
        [简介: {row['intro_philosophy'][:100]}]
        ---
        """

    # === C. 生成 ===
    combined_prompt = f"""
    【角色设定】
    你是一位世界级的鸡尾酒专家。
    
    【任务】
    根据顾客需求："{user_query}"
    从下面的【候选酒单】中挑选 3 款最合适的配方。
    
    【候选酒单】
    {context_text}

    【回复要求】
    1. 必须保留完整的原料用量和步骤。
    2. 中文回答，优雅专业。
    3. 格式：
       ### 🍸 [酒名]
       - **推荐理由**: ...
       - **原料**: ...
       - **步骤**: ...
    """

    try:
        print(f"正在请求模型: {MODEL_NAME}")
        
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[
                {"role": "user", "content": combined_prompt}
            ],
            temperature=0.7,
            max_tokens=4096, 
            presence_penalty=0.6
            # 🔴 注意：我删除了 extra_body 参数，因为 GPT 不需要它，也不会拦截酒精内容。
        )
        
        if not response.choices:
            return f"⚠️ API 返回空结果。请检查 Secrets 中的模型名称是否正确 (推荐 gpt-4o-mini)。", candidates
            
        return response.choices[0].message.content, candidates

    except Exception as e:
        return f"❌ AI 连接报错: {str(e)}", pd.DataFrame()
# --- 4. 界面 UI (保持不变) ---
# 这里为了美观，我们重新显示一下 Title，因为登录成功后才展示主界面
st.title("🍸 Punch AI 侍酒师")
st.caption(f"私人定制 · {MODEL_NAME}")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好，我是您的私人侍酒师。由于这是私人服务器，感谢您的邀请码验证。\n\n请告诉我您想喝点什么？"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("描述您的口味..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("AI 正在思考..."):
            ai_reply, related = get_ai_recommendation(prompt)
            st.markdown(ai_reply)
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})