# 我的书籍顾问 AI

一个可读取 PDF、TXT、EPUB，并结合书籍内容分析现实人际问题的 Streamlit 网页应用。

## 本地启动

1. 激活 `.venv`。
2. 确认 `.env` 中已配置 `OPENAI_API_KEY`。
3. 运行：`streamlit run app.py`

## 部署到 Streamlit Community Cloud

1. 把本项目上传到 GitHub 仓库。`.env` 不会上传，因为已经写入 `.gitignore`。
2. 在 Streamlit Community Cloud 新建应用，选择该仓库和 `app.py`。
3. 在应用的 Secrets 中，参照 `.streamlit/secrets.toml.example` 填写真实设置。
4. 部署完成后，把网页链接和邀请码发给体验者。

## 安全提醒

- 不要把 `.env`、真实密钥或 `secrets.toml` 发给别人或上传到 GitHub。
- 网页使用的是开发者的 API 额度，应设置邀请码和较小的调用上限。
- 当前上限是“单个浏览器会话”限制，不是严格的账户每日限制。正式公开前应增加登录、数据库计数和总预算告警。
