CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL,
    phone TEXT,
    pin_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS emotion_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL,
    emotions_json TEXT NOT NULL,
    intensity_json TEXT NOT NULL,
    rolling_scores_json TEXT NOT NULL,
    trend_json TEXT NOT NULL,
    energy_vx REAL,
    energy_vy REAL,
    energy_tau REAL,
    energy_angle REAL,
    energy_magnitude REAL,
    energy_coherence REAL,
    energy_angle_reliable INTEGER NOT NULL DEFAULT 0,
    energy_active_count INTEGER,
    emotion_source TEXT,
    emotion_error TEXT,
    energy_status TEXT,
    FOREIGN KEY (message_id) REFERENCES messages(id)
);

CREATE TABLE IF NOT EXISTS risk_flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    score REAL NOT NULL,
    keywords_json TEXT NOT NULL,
    threshold_hit INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    FOREIGN KEY (message_id) REFERENCES messages(id),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS doc_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_idx INTEGER NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents(id)
);

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,
    source_id INTEGER NOT NULL,
    vector_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assistant_message_meta (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    assistant_message_id INTEGER NOT NULL UNIQUE,
    response_json TEXT NOT NULL DEFAULT '{}',
    evidence_json TEXT NOT NULL DEFAULT '[]',
    schema_valid INTEGER NOT NULL DEFAULT 0,
    generation_source TEXT NOT NULL,
    sanitizer_hit INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (assistant_message_id) REFERENCES messages(id)
);

CREATE TABLE IF NOT EXISTS copilot_threads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    counselor_user_id INTEGER NOT NULL,
    selected_patient_id INTEGER,
    context_mode TEXT NOT NULL DEFAULT 'patient',
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (counselor_user_id) REFERENCES users(id),
    FOREIGN KEY (selected_patient_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS copilot_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    model TEXT,
    meta_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (thread_id) REFERENCES copilot_threads(id)
);

CREATE TABLE IF NOT EXISTS counselor_patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    counselor_user_id INTEGER NOT NULL,
    patient_user_id INTEGER NOT NULL UNIQUE,
    assigned_at TEXT NOT NULL,
    FOREIGN KEY (counselor_user_id) REFERENCES users(id),
    FOREIGN KEY (patient_user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS patient_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL UNIQUE,
    full_name TEXT NOT NULL,
    age INTEGER NOT NULL,
    gender TEXT NOT NULL CHECK (gender IN ('male', 'female', 'other', 'unknown')),
    residence TEXT,
    assistant_guidance TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    updated_by INTEGER,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (updated_by) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS counselor_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_user_id INTEGER NOT NULL,
    counselor_user_id INTEGER NOT NULL,
    session_summary TEXT NOT NULL,
    intervention_note TEXT,
    followup_plan TEXT,
    counselor_risk_level REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (patient_user_id) REFERENCES users(id),
    FOREIGN KEY (counselor_user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS patient_ai_consents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL UNIQUE,
    consent_given INTEGER NOT NULL DEFAULT 0,
    consent_version TEXT NOT NULL,
    consent_text TEXT NOT NULL,
    agreed_at TEXT,
    revoked_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    updated_by INTEGER,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (updated_by) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS app_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    updated_by INTEGER,
    FOREIGN KEY (updated_by) REFERENCES users(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_users_phone ON users(phone);
CREATE INDEX IF NOT EXISTS idx_copilot_threads_counselor ON copilot_threads(counselor_user_id);
CREATE INDEX IF NOT EXISTS idx_copilot_messages_thread ON copilot_messages(thread_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_counselor_patients_patient ON counselor_patients(patient_user_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_counselor_patients_pair ON counselor_patients(counselor_user_id, patient_user_id);
CREATE INDEX IF NOT EXISTS idx_counselor_patients_counselor ON counselor_patients(counselor_user_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_patient_profiles_user_id ON patient_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_counselor_notes_patient_created ON counselor_notes(patient_user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_counselor_notes_counselor_created ON counselor_notes(counselor_user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_patient_ai_consents_user_id ON patient_ai_consents(user_id);
CREATE INDEX IF NOT EXISTS idx_assistant_message_meta_message_id ON assistant_message_meta(assistant_message_id);
