import pickle
import os
import pandas as pd
from django.shortcuts import render
from django.conf import settings

# Create your views here.
# Đường dẫn tới các model đã lưu
# Django sẽ tìm từ thư mục gốc của dự án
MODEL_DIR = os.path.join(settings.BASE_DIR, 'ml_models', 'output_models')
LR_MODEL_PATH = os.path.join(MODEL_DIR, 'LogisticRegression.pkl')
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'RandomForest.pkl')
GB_MODEL_PATH = os.path.join(MODEL_DIR, 'GradientBoosting.pkl')

# Load model (chỉ load 1 lần khi server khởi động)
try:
    with open(RF_MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)
    print("Model loaded successfully from core/views.py!")
except Exception as e:
    print(f"Error loading model in core/views.py: {e}")
    pipeline = None

# View cho trang chủ (đổi tên từ index thành home_view)
def home_view(request):
    """
    Render trang chủ.
    """
    # context có thể truyền dữ liệu từ Python ra HTML
    context = {'page_title': 'Trang chủ'}
    return render(request, 'churn_predict/index.html', context)

# View cho trang phân tích (đổi tên từ analysis thành analysis_view)
def analysis_view(request):
    context = {'page_title': 'Phân tích dữ liệu'}
    return render(request, 'churn_predict/analysis.html', context)

# View cho trang giải thích mô hình (đổi tên từ explanation thành explanation_view)
def explanation_view(request):
    context = {'page_title': 'Giải thích mô hình'}
    return render(request, 'churn_predict/explanation.html', context)

# View cho trang dự đoán (đổi tên từ predict thành predict_view)
def predict_view(request):
    prediction_result = None
    prediction_proba = None
    form_data = request.POST if request.method == 'POST' else None
    if request.method == 'POST' and pipeline:
        try:
            # --- BẮT ĐẦU LOGIC DỰ ĐOÁN ---
            # Lấy dữ liệu từ form và chuyển đổi kiểu dữ liệu
            # Lưu ý: Tên 'name' trong thẻ <input> của HTML phải khớp với các key ở đây
            data = {
                'gender': request.POST.get('gender'),
                'SeniorCitizen': int(request.POST.get('SeniorCitizen', 0)),
                'Partner': request.POST.get('Partner'),
                'Dependents': request.POST.get('Dependents'),
                'tenure': int(request.POST.get('tenure', 0)),
                'PhoneService': request.POST.get('PhoneService'),
                'MultipleLines': request.POST.get('MultipleLines'),
                'InternetService': request.POST.get('InternetService'),
                'OnlineSecurity': request.POST.get('OnlineSecurity'),
                'OnlineBackup': request.POST.get('OnlineBackup'),
                'DeviceProtection': request.POST.get('DeviceProtection'),
                'TechSupport': request.POST.get('TechSupport'),
                'StreamingTV': request.POST.get('StreamingTV'),
                'StreamingMovies': request.POST.get('StreamingMovies'),
                'Contract': request.POST.get('Contract'),
                'PaperlessBilling': request.POST.get('PaperlessBilling'),
                'PaymentMethod': request.POST.get('PaymentMethod'),
                'MonthlyCharges': float(request.POST.get('MonthlyCharges', 0.0)),
                'TotalCharges': float(request.POST.get('TotalCharges', 0.0)),
            }

            # Tạo DataFrame với đúng thứ tự cột mà model đã được huấn luyện
            # (trừ cột 'churn')
            input_df = pd.DataFrame([data])

            # Dự đoán
            prediction = pipeline.predict(input_df)[0]
            prediction_proba = pipeline.predict_proba(input_df)[0][1] # Lấy xác suất của lớp 1 (Churn)

            # Chuyển kết quả thành dạng dễ đọc
            prediction_result = "Có" if prediction == 1 else "Không"
            prediction_proba = f"{prediction_proba:.2%}" # Format thành phần trăm

        except Exception as e:
            print(f"Error during prediction: {e}")
            prediction_result = f"Lỗi: {e}"
        # --- KẾT THÚC LOGIC DỰ ĐOÁN ---
    context = {
        'page_title': 'Dự đoán',
        'prediction': prediction_result,
        'probability': prediction_proba,
        'form_data': form_data
    }
    return render(request, 'churn_predict/predict.html', context)

# View cho trang giới thiệu (đổi tên từ about thành welcome_view)
def welcome_view(request):
    context = {'page_title': 'Chào mừng'}
    return render(request, 'churn_predict/welcome.html', context)
