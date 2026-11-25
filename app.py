import json
from html import escape
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from model import ARTIFACTS, create_cnn, create_mobilenet, extract_features, preprocess, read_image


# Konfigurasi halaman dan tampilan. Tidak memerlukan library antarmuka tambahan.
st.set_page_config(
    page_title="Clothing Classifier",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Palet warna dark mode. Data/hasil model tidak dipengaruhi CSS. */
    :root {
        color-scheme: dark;
        --bg: #0D1519; --card: #162229; --card-soft: #192A30;
        --card-deep: #111D23; --stroke: #2B4147; --stroke-hover: #4A6B66;
        --ink: #EAF4F0; --muted: #9AB4AB; --faint: #78938C;
        --accent: #68C6A4; --accent-strong: #2C7964;
        --track: #294047;
    }

    html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"],
    .stApp, .main { background: var(--bg) !important; color: var(--ink) !important; }
    [data-testid="stHeader"], [data-testid="stDecoration"] {
        background: transparent !important;
    }
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
    [data-testid="stToolbar"] { visibility: hidden; }
    .block-container { max-width: 1130px; padding-top: .85rem; padding-bottom: .8rem; }
    .stMarkdown, .stMarkdown p, [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"], [data-testid="stCheckbox"] label,
    [data-testid="stToggle"] label { color: var(--ink) !important; }

    /* Navigasi dua halaman tanpa sidebar. */
    .brand { display: flex; gap: 10px; align-items: center; padding: 3px 0 6px; }
    .brand-mark { width: 37px; height: 37px; border-radius: 11px;
        background: linear-gradient(145deg, #348C73, #205646); color: #F2FFF7;
        display:flex; align-items:center; justify-content:center;
        font-size:19px; font-weight:800; }
    .brand-name { color:var(--ink); font-size:17px; font-weight:780;
        letter-spacing:-.025em; line-height:1.1; }
    .brand-sub { font-size:11px; color:var(--muted); margin-top:4px; }
    .divider-line { height:1px; background:var(--stroke); margin:5px 0 11px; }
    .page-title { color:var(--ink); font-size:25px; font-weight:770;
        letter-spacing:-.035em; margin:0 0 3px; line-height:1.18; }
    .page-subtitle { color:var(--muted); font-size:12px; margin-bottom:12px; }
    .column-label { font-size:12px; color:#C2DCD1; font-weight:750;
        letter-spacing:.015em; margin:0 0 6px; }

    /* Kolom preview: ikon SVG jelas, bukan simbol seperti gambar rusak. */
    .photo-blank { height:258px; border:1px dashed var(--stroke-hover);
        background:var(--card-deep); display:flex; flex-direction:column;
        align-items:center; justify-content:center; gap:10px;
        border-radius:14px; color:var(--muted); }
    .photo-blank .photo-icon { width:54px; height:54px; color:var(--accent); opacity:.85; }
    .photo-blank strong { font-size:13px; font-weight:700; color:var(--ink); }
    .photo-blank span { font-size:12px; color:var(--muted); }
    .photo-label { text-align:center; color:var(--muted); font-size:11px; margin:4px 0 0; }

    /* Hasil kosong mengisi area kanan agar seimbang dengan upload + foto. */
    .result-empty { border:1px solid var(--stroke); background:var(--card);
        border-radius:14px; padding:24px 23px; height:337px;
        display:flex; flex-direction:column; align-items:flex-start;
        justify-content:center; gap:9px; color:var(--muted); }
    .result-empty .empty-mark { font-size:25px; line-height:1; color:var(--accent); margin-bottom:2px; }
    .result-empty strong { color:var(--ink); font-size:15px; }
    .result-empty span { font-size:12px; line-height:1.55; }

    /* Hasil prediksi singkat: semua 8 probabilitas tetap muat di satu layar. */
    .result { border:1px solid #366050;
        background:linear-gradient(120deg,#1A332F 0%,#15252A 100%);
        border-radius:14px; padding:13px 17px; margin-bottom:8px; }
    .result small { color:#9FDFC4; font-size:10px; letter-spacing:.11em;
        font-weight:750; text-transform:uppercase; }
    .result .name { color:#DCFDEB; font-size:27px; letter-spacing:-.04em;
        font-weight:790; margin:3px 0 3px; line-height:1.08; }
    .result .conf { color:#ADD3C1; font-size:12px; }
    .prob-wrap { border:1px solid var(--stroke); background:var(--card);
        border-radius:13px; padding:10px 13px; }
    .prob-heading { font-size:12px; color:var(--ink); font-weight:750; margin-bottom:7px; }
    .prob-row { display:grid; grid-template-columns:89px 1fr 45px; gap:9px;
        align-items:center; margin:5px 0; min-height:13px; }
    .prob-name { color:#C6D9D1; font-size:11px; }
    .prob-track { background:var(--track); border-radius:99px;
        height:7px; overflow:hidden; }
    .prob-fill { background:#648F83; border-radius:99px; height:7px; }
    .prob-fill.top { background:var(--accent); }
    .prob-value { color:#B1CFC2; font-size:11px; text-align:right;
        font-variant-numeric:tabular-nums; }
    .mini-label { font-size:10px; color:var(--muted); margin-top:6px; line-height:1.5; }
    .compare-row { display:flex; align-items:center; justify-content:space-between;
        padding:7px 10px; border:1px solid var(--stroke);
        border-radius:9px; background:var(--card); margin:5px 0; gap:10px; }
    .compare-name { color:var(--ink); font-size:12px; font-weight:700; }
    .compare-answer { color:var(--accent); font-size:12px; text-align:right; }

    /* Halaman Ringkasan eksperimen. */
    .metric-row { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }
    .metric { background:var(--card); border:1px solid var(--stroke);
        border-radius:13px; padding:16px 17px; }
    .metric .label { color:var(--muted); font-size:11px; margin-bottom:7px; }
    .metric .number { color:#A7E4CB; font-size:25px;
        font-weight:770; letter-spacing:-.04em; }
    .stat-title { font-size:15px; color:var(--ink); font-weight:750; margin:17px 0 8px; }
    .info-bar { background:var(--card); border:1px solid var(--stroke);
        color:#BBD3C7; border-radius:10px; padding:11px 14px;
        font-size:12px; line-height:1.7; }
    .st-key-confusion_matrix_button { padding-top: 12px !important; }
    .compare-table { width:100%; border-collapse:collapse; background:var(--card);
        border:1px solid var(--stroke); border-radius:12px; overflow:hidden; }
    .compare-table th { text-align:left; background:#20343A; color:#D2E7DE;
        font-size:12px; padding:11px 15px; }
    .compare-table td { color:var(--ink); font-size:13px; padding:11px 15px;
        border-top:1px solid var(--stroke); }

    /* Uploader kosong dibuat rata tengah, tanpa menggeser tombol secara manual. */
    [data-testid="stFileUploaderDropzone"] {
        background:var(--card) !important;
        border:1px dashed var(--stroke-hover) !important;
        border-radius:12px !important;
        padding:10px 14px !important;
        min-height:64px !important;
        display:flex !important;
        align-items:center !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover { border-color:var(--accent) !important; }
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] div { color:var(--muted) !important; opacity:1 !important; }
    [data-testid="stFileUploaderDropzone"] button {
        background:#2C725D !important; color:#F5FFF9 !important;
        border-radius:9px !important; border:0 !important;
        position:static !important; transform:none !important; margin:0 !important;
    }
    [data-testid="stFileUploaderDropzone"] button * { color:#F5FFF9 !important; }

    /* Setelah upload, cukup tampilkan kontrol kecil di atas preview foto. */
    .st-key-replace_photo button,
    .st-key-remove_photo button {
        box-sizing: border-box !important;
        min-height: 34px !important;
        height: auto !important;
        min-width: 0 !important;
        padding: 6px 8px !important;
        font-size: 12px !important;
        font-weight: 650 !important;
        border-radius: 8px !important;
        width: 100% !important;
        white-space: normal !important;
        line-height: 1.3 !important;
        text-align: center !important;
        box-shadow: none !important;
    }
    /* Streamlit renders the button caption as a nested Markdown paragraph. */
    .st-key-replace_photo button p,
    .st-key-remove_photo button p {
        font-size: 12px !important;
        font-weight: 650 !important;
        line-height: 1.3 !important;
        margin: 0 !important;
        white-space: normal !important;
        overflow-wrap: normal !important;
        text-align: center !important;
    }
    .st-key-replace_photo button [data-testid="stMarkdownContainer"],
    .st-key-remove_photo button [data-testid="stMarkdownContainer"] {
        min-width: 0 !important;
        max-width: 100% !important;
    }
    .st-key-remove_photo button { color: #E4A6A0 !important; }
    .st-key-remove_photo button:hover {
        color: #FFC0B6 !important;
        border-color: #AC6B65 !important;
        background: #2D242A !important;
    }
    /* Hindari jarak ekstra pada row toolbar ketika foto sudah ada. */
    [data-testid="stHorizontalBlock"]:has(.st-key-replace_photo) {
        align-items: center !important;
    }
    [data-testid="stImage"] img { max-height:258px; width:auto; object-fit:contain;
        border-radius:12px; margin:0 auto; }
    div[data-testid="stButton"] button { border-radius:10px;
        font-weight:650; min-height:36px; padding:.36rem .7rem;
        transition:background .15s ease,border-color .15s ease; }
    div[data-testid="stButton"] button[kind="primary"] {
        background:#2C725D !important; border-color:#3F967A !important;
        color:#F6FFF9 !important; }
    div[data-testid="stButton"] button[kind="primary"]:hover { background:#37876F !important; }
    div[data-testid="stButton"] button[kind="secondary"] {
        background:var(--card) !important; border:1px solid var(--stroke) !important;
        color:var(--ink) !important; }
    div[data-testid="stButton"] button[kind="secondary"]:hover {
        background:#20373A !important; border-color:var(--accent) !important; }
    [data-testid="stCheckbox"] label,
    [data-testid="stToggle"] label,
    [data-testid="stCaptionContainer"] { color:var(--muted) !important; }
    [data-testid="stExpander"] { background:var(--card) !important;
        border:1px solid var(--stroke) !important; }

    @media (max-width:850px) {
        .block-container { padding-top:.8rem; }
        .metric-row { grid-template-columns:1fr; }
        .photo-blank { height:218px; }
        .result-empty { height:218px; }
        .page-title { font-size:23px; }
        .brand-name { font-size:15px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Cache model: tidak dibaca ulang setiap kali pengguna mengubah kontrol halaman.
@st.cache_resource(show_spinner=False)
def load_cnn(model_version):
    metadata = json.loads((ARTIFACTS / "experiment.json").read_text(encoding="utf-8"))
    labels = list(metadata["classes"])
    network = create_cnn(len(labels))
    network.load_state_dict(
        torch.load(ARTIFACTS / "cnn.pt", map_location="cpu", weights_only=True)
    )
    network.eval()
    return network


@st.cache_resource(show_spinner=False)
def load_other_models(model_version):
    encoder = create_mobilenet(pretrained=False)
    encoder.load_state_dict(
        torch.load(ARTIFACTS / "encoder.pt", map_location="cpu", weights_only=True)
    )
    encoder.eval()
    logistic = joblib.load(ARTIFACTS / "logistic.joblib")
    forest = joblib.load(ARTIFACTS / "forest.joblib")
    return encoder, logistic, forest


# Prediksi menggunakan preprocessing evaluasi, bukan augmentasi training.
@torch.inference_mode()
def predict_cnn(image, network):
    tensor = preprocess(training=False)(read_image(image)).unsqueeze(0)
    probability = torch.softmax(network(tensor), dim=1)[0].cpu().numpy()
    return tensor, probability


@torch.inference_mode()
def predict_other_models(tensor, classes, encoder, logistic, forest):
    features = extract_features(encoder, tensor).cpu().numpy()
    answers = {}
    for name, model in (("Logistic Regression", logistic), ("Random Forest", forest)):
        scores = np.zeros(len(classes), dtype=float)
        scores[np.asarray(model.classes_, dtype=int)] = model.predict_proba(features)[0]
        answers[name] = scores
    return answers


def probability_html(classes, values):
    ranked = sorted(zip(classes, values), key=lambda pair: float(pair[1]), reverse=True)
    rows = []
    for index, (name, score) in enumerate(ranked):
        percent = max(0.0, min(100.0, float(score) * 100))
        rows.append(
            f'<div class="prob-row"><span class="prob-name">{escape(name)}</span>'
            f'<div class="prob-track"><div class="prob-fill {"top" if index == 0 else ""}" '
            f'style="width:{percent:.2f}%"></div></div>'
            f'<span class="prob-value">{percent:.1f}%</span></div>'
        )
    return '<div class="prob-wrap"><div class="prob-heading">Probabilitas tiap kelas</div>' + "".join(rows) + '</div>'


def summary_metrics_html(metadata):
    outcome = metadata.get("test_results_selected_model", {})
    def score(key):
        value = outcome.get(key)
        return f"{float(value):.2%}" if value is not None else "—"
    items = (("Model terbaik (validation)", str(metadata.get("validation_winner", "—"))),
             ("Accuracy test", score("accuracy")), ("Macro F1 test", score("macro_f1")))
    return '<div class="metric-row">' + "".join(
        f'<div class="metric"><div class="label">{escape(label)}</div>'
        f'<div class="number">{escape(value)}</div></div>' for label, value in items
    ) + '</div>'


def comparison_table_html(frame):
    cols = ['model', 'accuracy', 'macro_f1']
    if not set(cols).issubset(frame.columns):
        return None
    rows = []
    for _, item in frame.sort_values('macro_f1', ascending=False).iterrows():
        rows.append('<tr><td>' + escape(str(item['model'])) + '</td><td>'
                    + f"{float(item['accuracy']):.2%}" + '</td><td>'
                    + f"{float(item['macro_f1']):.2%}" + '</td></tr>')
    return ('<table class="compare-table"><thead><tr><th>Model</th><th>Accuracy</th>'
            '<th>Macro F1</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def go_to(page):
    st.session_state.page = page


# Dua halaman dalam satu file, agar struktur proyek tetap sederhana.
if 'page' not in st.session_state:
    st.session_state.page = 'prediksi'
if 'photo_bytes' not in st.session_state:
    st.session_state.photo_bytes = None
if 'upload_rev' not in st.session_state:
    st.session_state.upload_rev = 0
if 'photo_name' not in st.session_state:
    st.session_state.photo_name = 'foto_pakaian'
if 'photo_size' not in st.session_state:
    st.session_state.photo_size = 0

nav_brand, nav_predict, nav_summary = st.columns([5.5, 1.25, 1.65], gap='small')
with nav_brand:
    st.markdown(
        '<div class="brand"><span class="brand-mark">C</span><div>'
        '<div class="brand-name">Clothing Classifier</div>'
        '<div class="brand-sub">Klasifikasi pakaian · 8 kategori</div></div></div>',
        unsafe_allow_html=True,
    )
with nav_predict:
    st.button('Prediksi', use_container_width=True,
              type='primary' if st.session_state.page == 'prediksi' else 'secondary',
              on_click=go_to, args=('prediksi',))
with nav_summary:
    st.button('Ringkasan eksperimen', use_container_width=True,
              type='primary' if st.session_state.page == 'eksperimen' else 'secondary',
              on_click=go_to, args=('eksperimen',))
st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)

info_path = ARTIFACTS / 'experiment.json'
if not info_path.exists():
    st.warning('Hasil training belum tersedia. Jalankan `python train.py` terlebih dahulu.')
    st.stop()

try:
    metadata = json.loads(info_path.read_text(encoding='utf-8'))
    classes = list(metadata['classes'])
except Exception as error:
    st.error(f'Gagal membaca ringkasan training: {error}')
    st.stop()

if st.session_state.page == 'eksperimen':
    # Halaman terpisah: informasi eksperimen tidak memenuhi halaman prediksi.
    st.markdown('<div class="page-title">Ringkasan eksperimen</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Hasil berdasarkan laporan training yang tersimpan, bukan angka contoh.</div>', unsafe_allow_html=True)
    st.markdown(summary_metrics_html(metadata), unsafe_allow_html=True)

    st.markdown('<div class="stat-title">Perbandingan model pada validation</div>', unsafe_allow_html=True)
    comparison_file = ARTIFACTS / 'validation_comparison.csv'
    if comparison_file.exists():
        try:
            table_html = comparison_table_html(pd.read_csv(comparison_file))
            if table_html:
                st.markdown(table_html, unsafe_allow_html=True)
            else:
                st.info('Format laporan validation tidak sesuai.')
        except Exception as error:
            st.info(f'Laporan validation tidak dapat dibaca: {error}')
    else:
        st.info('Laporan validation belum tersedia.')

    st.markdown('<div class="stat-title">Data dan evaluasi</div>', unsafe_allow_html=True)
    split_counts = metadata.get('split_counts', {})
    train_n = split_counts.get('train', '—')
    val_n = split_counts.get('validation', '—')
    test_n = split_counts.get('test', '—')
    st.markdown(
        f'<div class="info-bar"><b>{escape(str(metadata.get("n_samples", "—")))} gambar · '
        f'{len(classes)} kelas</b> &nbsp;|&nbsp; Training: {escape(str(train_n))}'
        f' &nbsp;·&nbsp; Validation: {escape(str(val_n))}'
        f' &nbsp;·&nbsp; Testing: {escape(str(test_n))}<br>'
        'Input RGB 160 × 160, normalisasi ImageNet. Model dipilih berdasarkan Macro F1 validation.' '</div>',
        unsafe_allow_html=True,
    )

    matrix_file = ARTIFACTS / 'test_confusion_matrix.png'

    @st.dialog('Confusion matrix — data testing')
    def show_confusion_matrix():
        st.image(str(matrix_file), use_container_width=True)
        st.caption('Baris adalah label sebenarnya; kolom merupakan prediksi model.')

    if matrix_file.exists():
        if st.button('Lihat confusion matrix', key='confusion_matrix_button'):
            show_confusion_matrix()
    st.caption('Pengujian akhir tidak digunakan untuk memilih model terbaik.')

else:
    # Halaman utama: upload, foto, prediksi dan probabilitas dalam satu tampilan.
    st.markdown('<div class="page-title">Klasifikasi jenis pakaian</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Unggah foto satu jenis pakaian untuk melihat hasil prediksi.</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1.05], gap='large')
    with left:
        # Setelah foto diunggah, nama file panjang tidak ditampilkan.
        # Preview langsung menjadi fokus, sedangkan Ganti/Hapus tetap mudah dijangkau.
        if st.session_state.photo_bytes is None:
            st.markdown('<div class="column-label">Foto pakaian</div>', unsafe_allow_html=True)
            upload = st.file_uploader(
                'Pilih gambar', type=['jpg', 'jpeg', 'png', 'webp'],
                label_visibility='collapsed',
                key=f'photo_upload_{st.session_state.upload_rev}',
            )
            if upload is not None:
                st.session_state.photo_bytes = upload.getvalue()
                st.session_state.photo_name = upload.name
                st.session_state.photo_size = len(st.session_state.photo_bytes)
                st.rerun()
        else:
            # Kontrol menyatu dengan judul kolom, tanpa kartu status file yang memakan tempat.
            label_col, change_col, clear_col = st.columns(
                [3.0, 2.4, 1.6], gap='small', vertical_alignment='center'
            )
            with label_col:
                st.markdown('<div class="column-label">Foto pakaian</div>', unsafe_allow_html=True)
            with change_col:
                if st.button('Ganti foto', use_container_width=True, key='replace_photo',
                             help='Pilih gambar pakaian lain'):
                    st.session_state.photo_bytes = None
                    st.session_state.upload_rev += 1
                    st.rerun()
            with clear_col:
                if st.button('Hapus', use_container_width=True, key='remove_photo',
                             help='Hapus gambar dari tampilan'):
                    st.session_state.photo_bytes = None
                    st.session_state.photo_name = 'foto_pakaian'
                    st.session_state.photo_size = 0
                    st.session_state.upload_rev += 1
                    st.rerun()
        photo = None
        if st.session_state.photo_bytes is not None:
            try:
                photo = read_image(Image.open(BytesIO(st.session_state.photo_bytes)))
                st.image(photo, use_container_width=True)
            except Exception as error:
                st.error(f'File gambar gagal dibuka: {error}')
                st.session_state.photo_bytes = None
        else:
            st.markdown(
                '<div class="photo-blank">'
                '<svg class="photo-icon" viewBox="0 0 48 48" fill="none" '
                'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
                '<rect x="7" y="9" width="34" height="30" rx="5" '
                'stroke="currentColor" stroke-width="2"/>'
                '<circle cx="18" cy="19" r="3" fill="currentColor"/>'
                '<path d="M10 33l10-9 7 6 5-5 7 8" stroke="currentColor" '
                'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
                '</svg>'
                '<strong>Belum ada gambar</strong><span>JPG, PNG, atau WEBP</span></div>',
                unsafe_allow_html=True,
            )

    with right:
        st.markdown('<div class="column-label">Hasil prediksi</div>', unsafe_allow_html=True)
        if photo is None:
            st.markdown(
                '<div class="result-empty"><div class="empty-mark">✦</div>'
                '<strong>Siap untuk memprediksi</strong>'
                '<span>Unggah satu foto pakaian untuk melihat kategori dan '
                'skor prediksi dari model.</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="mini-label">Model utama dipilih berdasarkan nilai Macro F1 validation.</div>',
                        unsafe_allow_html=True)
        else:
            try:
                if not (ARTIFACTS / 'cnn.pt').exists():
                    st.error('Bobot CNN belum tersedia. Jalankan training terlebih dahulu.')
                    st.stop()
                version = (ARTIFACTS / 'cnn.pt').stat().st_mtime_ns
                with st.spinner('Memproses gambar...'):
                    cnn = load_cnn(version)
                    tensor, cnn_prob = predict_cnn(photo, cnn)
                    winner = metadata.get('validation_winner', 'CNN')
                    probabilities = cnn_prob
                    predictions = {'CNN': cnn_prob}
                    if winner != 'CNN':
                        needed = ['encoder.pt', 'logistic.joblib', 'forest.joblib']
                        if any(not (ARTIFACTS / name).exists() for name in needed):
                            st.error('Bobot model terbaik belum lengkap.')
                            st.stop()
                        versions = tuple((ARTIFACTS / name).stat().st_mtime_ns for name in needed)
                        encoder, logistic, forest = load_other_models(versions)
                        predictions.update(predict_other_models(tensor, classes, encoder, logistic, forest))
                        probabilities = predictions[winner]

                idx = int(np.argmax(probabilities))
                st.markdown(
                    '<div class="result"><small>Prediksi · ' + escape(winner) + '</small>'
                    '<div class="name">' + escape(classes[idx]) + '</div>'
                    '<div class="conf">Skor model: ' + f'{float(probabilities[idx]):.1%}' + '</div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(probability_html(classes, probabilities), unsafe_allow_html=True)
                st.markdown('<div class="mini-label">Skor probabilitas bukan jaminan prediksi benar.</div>',
                            unsafe_allow_html=True)

                # Model pembanding hanya dimuat jika diminta untuk mengurangi beban awal PC.
                if st.toggle('Bandingkan prediksi 3 model', value=False):
                    needed = ['encoder.pt', 'logistic.joblib', 'forest.joblib']
                    if any(not (ARTIFACTS / name).exists() for name in needed):
                        st.info('Model pembanding belum tersedia.')
                    else:
                        versions = tuple((ARTIFACTS / name).stat().st_mtime_ns for name in needed)
                        encoder, logistic, forest = load_other_models(versions)
                        if len(predictions) == 1:
                            predictions.update(predict_other_models(tensor, classes, encoder, logistic, forest))
                        for name in ['CNN', 'Logistic Regression', 'Random Forest']:
                            scores = predictions[name]
                            pos = int(np.argmax(scores))
                            st.markdown(
                                '<div class="compare-row"><span class="compare-name">' + escape(name) + '</span>'
                                '<span class="compare-answer">' + escape(classes[pos])
                                + f' · {float(scores[pos]):.1%}' + '</span></div>',
                                unsafe_allow_html=True,
                            )
            except Exception as error:
                st.error(f'Prediksi gagal: {error}')
