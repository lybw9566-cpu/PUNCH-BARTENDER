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
DATA_FILE = "punch_recipes.jsonl"

st.set_page_config(page_title="Punch AI 侍酒师", page_icon="🍸", layout="wide") 
# 注意：layout 改为 'wide' 可以让侧边栏和主内容更宽敞

# --- 2. 数据加载与向量化 (保持 char_wb 模糊搜索) ---
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

    # 混合文本用于搜索
    df['combined_text'] = (
        df['title'].fillna('') + " " + 
        df['ingredients'].astype(str) + " " + 
        df['tags'].astype(str)
    )

    # 使用字符级 n-gram 实现模糊匹配
    vectorizer = TfidfVectorizer(
        stop_words='english',
        analyzer='char_wb', 
        ngram_range=(3, 5)
    )
    
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

    return df, vectorizer, tfidf_matrix

df, vectorizer, tfidf_matrix = load_data_and_vectors()

if df is None:
    st.error(f"❌ 找不到数据文件 {DATA_FILE}")
    st.stop()

# --- 3. 核心 AI 逻辑 ---
def get_ai_recommendation(user_query):
    # === A. 检索 ===
    try:
        user_vec = vectorizer.transform([user_query])
        similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-30:][::-1]
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
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "user", "content": combined_prompt}],
            temperature=0.7,
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

st.title("🍸 Punch AI 侍酒师")

# --- 🔍 侧边栏：超级模糊搜索 ---
with st.sidebar:
    st.header("📖 配方百科全书")
    # 1. 搜索框
    search_query = st.text_input("🔍 搜索配方 (支持模糊拼写)", placeholder="例如: Bronx 或 margrita")
    
    selected_recipe_id = None
    
    if search_query:
        # 复用那个强大的向量搜索引擎
        # 即使你输错 "Mrgarita"，它也能算出它是 Margarita
        search_vec = vectorizer.transform([search_query])
        sims = cosine_similarity(search_vec, tfidf_matrix).flatten()
        
        # 找出最相似的 10 个
        top_indices = sims.argsort()[-10:][::-1]
        
        # 制作下拉菜单选项字典: { "酒名": ID }
        options_map = {}
        for i in top_indices:
            row = df.iloc[i]
            # 如果相似度太低(小于0.1)，可能是噪音，不显示
            if sims[i] > 0.1:
                options_map[f"{row['title']}"] = i
        
        if options_map:
            st.success(f"找到 {len(options_map)} 个相关结果:")
            # 2. 下拉选择框
            selected_name = st.selectbox("👇 点击选择查看详情:", list(options_map.keys()))
            
            if selected_name:
                selected_recipe_id = options_map[selected_name]
        else:
            st.warning("🤔 未找到相似配方，请换个词试试")

# --- 📋 主界面：展示配方详情卡片 (如果有选中) ---
if selected_recipe_id is not None:
    # 获取选中行的数据
    recipe_data = df.iloc[selected_recipe_id]
    
    # 渲染卡片容器
    with st.container(border=True):
        col_close, col_title = st.columns([1, 8])
        with col_title:
            st.header(f"🍹 {recipe_data['title']}")
        
        # 显示简介
        st.info(f"💡 {recipe_data['intro_philosophy']}")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🧂 原料 Ingredients")
            # 处理原料列表显示
            ingredients_list = recipe_data['ingredients']
            if isinstance(ingredients_list, str):
                st.write(ingredients_list)
            elif isinstance(ingredients_list, list):
                for ing in ingredients_list:
                    st.markdown(f"- {ing}")
                    
        with c2:
            st.subheader("🥣 做法 Instructions")
            st.write(recipe_data['instructions'])
            
        st.caption(f"Tags: {recipe_data.get('tags', 'Classic')}")
        
    st.markdown("---") # 分割线，下面是聊天区

# --- 💬 聊天区域 (AI 侍酒师) ---
st.caption(f"私人定制 · {MODEL_NAME}")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好！您可以在左侧搜索特定的配方卡片，也可以直接在这里告诉我您的口味，让我为您推荐。"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("描述您的口味，或让 AI 推荐..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("AI 正在思考..."):
            ai_reply, related = get_ai_recommendation(prompt)
            st.markdown(ai_reply)
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})