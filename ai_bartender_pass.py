import random  # <--- 1. 引入随机库，用于打破推荐的重复性
import streamlit as st
import pandas as pd
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 0. 门卫系统 ---
def check_access():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔒 这是一个私密应用")
    st.write("请在下方输入邀请码以继续访问。")
    
    user_input = st.text_input("请输入邀请码 (Access Key)", type="password")
    
    if st.button("解锁进入"):
        valid_keys = st.secrets.get("access_keys", [])
        if user_input in valid_keys:
            st.session_state.authenticated = True
            st.success("✅ 验证成功！正在加载...")
            st.rerun()
        else:
            st.error("❌ 邀请码无效")
    return False

if not check_access():
    st.stop()

# --- 1. 配置加载 ---
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

# 🔴 2. 这里的数据库文件必须是翻译好的中文版
DATA_FILE = "punch_recipes_cn.jsonl"

st.set_page_config(page_title="Punch AI 调酒师", page_icon="🍸", layout="wide") 

# --- 2. 数据加载与向量化 ---
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

    # 混合文本用于搜索 (包含标题、原料、标签)
    df['combined_text'] = (
        df['title'].fillna('') + " " + 
        df['ingredients'].astype(str) + " " + 
        df['tags'].astype(str)
    )

    # 使用字符级 n-gram 实现模糊匹配
    # 即使数据库是中文，保留这个设置也能很好地匹配英文酒名
    vectorizer = TfidfVectorizer(
        stop_words='english',
        analyzer='char_wb', 
        ngram_range=(3, 5),
        max_features=5000 # 限制特征数量，防止加载过慢
    )
    
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

    return df, vectorizer, tfidf_matrix

df, vectorizer, tfidf_matrix = load_data_and_vectors()

if df is None:
    st.error(f"❌ 找不到数据文件 {DATA_FILE}")
    st.stop()

# --- 3. 核心 AI 逻辑 (已加入“鱼塘扩容”逻辑) ---
def get_ai_recommendation(user_query):
    # === A. 检索 ===
    try:
        user_vec = vectorizer.transform([user_query])
        similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
        
        # 🔴 3. 扩大候选池 (鱼塘逻辑)
        # 取前 100 个相关的结果
        top_k = 100 
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # 🔴 4. 随机洗牌 (Shuffling)
        # 从这 100 个里随机抽 20 个，打破“总是推荐第一名”的魔咒
        candidates_pool = top_indices.tolist()
        
        if len(candidates_pool) > 20:
            selected_indices = random.sample(candidates_pool, 20)
        else:
            selected_indices = candidates_pool
            
        candidates = df.iloc[selected_indices]

    except Exception as e:
        return f"检索系统出错了: {e}", pd.DataFrame()

    # === B. 增强 (构建 Context) ===
    context_text = ""
    for idx, row in candidates.iterrows():
        # 直接读取中文数据
        context_text += f"""
        [酒名: {row['title']}]
        [原料: {row['ingredients']}]
        [步骤: {row['instructions']}]
        [简介: {row.get('intro_philosophy', '')[:100]}]
        ---
        """

    # === C. 生成 (Prompt) ===
    combined_prompt = f"""
    【角色设定】
    你是一位见多识广的调酒师，擅长发掘冷门佳酿。
    
    【任务】
    用户想喝："{user_query}"
    从下面的【候选酒单】中，挑选 3 款推荐给用户。
    
    【策略要求】
    1. **不要总是推荐最常见的酒**。如果候选名单里有独特、冷门但符合用户口味的配方，优先推荐它们，给用户惊喜。
    2. 如果有多种基酒选择，请展示多样性。
    3. 基于提供的数据直接回答，因为数据已经是中文的了。
    
    【候选酒单】
    {context_text}

    【回复格式】
    ### 🍸 [酒名] (保持英文原名)
    - **推荐理由**: ...
    - **原料**: ...
    - **步骤**: ...
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "user", "content": combined_prompt}],
            temperature=0.8, 
            max_tokens=4096, 
            presence_penalty=0.6 
        )
        if not response.choices:
            return f"⚠️ API 返回空结果。", candidates
        return response.choices[0].message.content, candidates

    except Exception as e:
        return f"❌ AI 连接报错: {str(e)}", pd.DataFrame()

# ==========================================
# 🎨 界面布局开始
# ==========================================

st.title("🍸 Punch AI 调酒师")

# --- 🔍 侧边栏：配方百科 (无翻译模块，直接显示) ---
with st.sidebar:
    st.header("📖 配方百科全书")
    search_query = st.text_input("🔍 搜索配方 (支持模糊拼写)", placeholder="例如: Bronx")
    
    selected_recipe_id = None
    
    if search_query:
        search_vec = vectorizer.transform([search_query])
        sims = cosine_similarity(search_vec, tfidf_matrix).flatten()
        
        # 找出最相似的 10 个
        top_indices = sims.argsort()[-10:][::-1]
        
        options_map = {}
        for i in top_indices:
            row = df.iloc[i]
            if sims[i] > 0.1:
                options_map[f"{row['title']}"] = i
        
        if options_map:
            st.success(f"找到 {len(options_map)} 个相关结果:")
            selected_name = st.selectbox("👇 点击选择查看详情:", list(options_map.keys()))
            
            if selected_name:
                selected_recipe_id = options_map[selected_name]
        else:
            st.warning("🤔 未找到相似配方")

# ==========================================
# 📋 主界面：配方详情卡片 (静态显示，无需AI翻译)
# ==========================================
if selected_recipe_id is not None:
    # 🔴 5. 直接读取数据库里的中文数据
    recipe_data = df.iloc[selected_recipe_id]
    
    with st.container(border=True):
        col_close, col_title = st.columns([1, 9])
        
        # 关闭按钮
        with col_close:
            if st.button("❌", key="close_btn"):
                selected_recipe_id = None
                st.rerun()

        with col_title:
            st.header(f"🍹 {recipe_data['title']}") # 标题保持英文
        
        # 简介 (数据库里已经是中文了)
        st.info(f"💡 {recipe_data.get('intro_philosophy', '暂无简介')}")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🧂 原料 Ingredients")
            # 兼容处理：如果是列表直接显示，如果是字符串则直接显示
            ings = recipe_data['ingredients']
            if isinstance(ings, list):
                for ing in ings:
                    st.write(f"• {ing}")
            else:
                st.write(ings)
                    
        with c2:
            st.subheader("🥣 做法 Instructions")
            st.write(recipe_data['instructions'])
            
        st.caption(f"标签: {recipe_data.get('tags', 'Classic')}")
        
    st.markdown("---") 

# --- 💬 聊天区域 ---
st.caption(f"私人定制 · {MODEL_NAME}")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好！我是您的 AI 侍酒师。您可以直接点餐，或者在左侧查阅配方。"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("今天想喝点什么风味的？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("正在配方库中搜寻..."):
            ai_reply, related = get_ai_recommendation(prompt)
            st.markdown(ai_reply)
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})