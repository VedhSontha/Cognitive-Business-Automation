# Development & Contribution Guide

This guide provides architectural details and coding standards for developers contributing to the Cognitive Business Automation Suite.

## 🛠️ Streamlit Local Run Best Practices
1. **Caching Large Models:** When loading trained model models (e.g., `scaler.pkl`, `best_model.pkl`), use Streamlit's cache mechanism (`@st.cache_resource` or `@st.cache_data`) to prevent reloading the weights on every user interaction or widget state refresh.
2. **Session State Management:** Use `st.session_state` to store state data (e.g., loaded invoice objects in FlowbitAI or dialogue history in Task 3) across page updates.
3. **Port Conflicts:** If multiple Streamlit apps are running, specify a custom port:
   ```bash
   streamlit run app.py --server.port 8502
   ```

## 🗄️ Database Guidelines (FlowbitAI SQLite)
* **Local SQLite State:** The database `flowbit.db` resides locally inside `FlowbitAI`. Do not commit actual parsed transaction data containing private invoices.
* **Database Reset:** You can reset or re-initialize the database schemas at any time by running the SQL initialization script:
  ```bash
  python -m src.init_db
  ```
* **Safe Migrations:** If modifying schema structure (e.g. adding new columns for PO matches), update `src/models.py` and run a migration script instead of manually editing raw DB tables.
