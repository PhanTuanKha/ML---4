import time
from django.templatetags.static import static
import pickle
import os
import pandas as pd
from django.shortcuts import render, redirect # --- SỬA ĐỔI --- (Thêm 'redirect')
from django.conf import settings
import numpy as np # <-- MỚI
import json
import sys # <-- MỚI
import subprocess # <-- MỚI
from django.contrib import messages # <-- MỚI
from django.core.files.storage import default_storage # <-- MỚI
import time
from django.templatetags.static import static

# --- MỚI: Thêm các thư viện cho logic Auth ---
from django.http import JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt # Sẽ dùng cho API
from .models import User # Import Model 'User' chúng ta đã tạo
from .models import (
    User, Contracts, Customers, InternetServices, 
    PaymentMethods, PhoneServices, ChurnRecords
)
try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    LimeTabularExplainer = None
    print("Lỗi: Thư viện LIME chưa được cài đặt. Chạy 'pip install lime'")

# Create your views here.
# Đường dẫn tới các model đã lưu
# (Code load model ML của bạn giữ nguyên)
MODEL_DIR = os.path.join(settings.BASE_DIR, 'ml_models', 'output_models')
LR_MODEL_PATH = os.path.join(MODEL_DIR, 'LogisticRegression.pkl')
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'RandomForest.pkl')
GB_MODEL_PATH = os.path.join(MODEL_DIR, 'GradientBoosting.pkl')


# --- SỬA ĐỔI: Đổi tên 'home_view' thành 'index_view' ---
# View này render file index.html (chứa form đăng nhập/đăng ký)
def index_view(request):
    """
    Render trang chủ (index.html).
    """
    context = {'page_title': 'Trang chủ'}
    return render(request, 'churn_predict/index.html', context)

# --- SỬA ĐỔI: Cập nhật 'welcome_view' để kiểm tra đăng nhập ---
# View này là trang 'welcome.html', chỉ xem được SAU KHI đăng nhập
def welcome_view(request):
    # Kiểm tra xem 'user_id' có trong session không
    if 'user_id' not in request.session:
        # Nếu chưa đăng nhập, đá về trang chủ (trang đăng nhập)
        return redirect('index_view') 

    # --- SỬA ĐỔI: Lấy tên đầy đủ (user_full_name) từ session ---
    # Lấy tên, nếu không có thì dự phòng là chữ 'User'
    user_name = request.session.get('user_full_name', 'User') 
    
    context = {
        'page_title': 'Chào mừng',
        'user_name': user_name # Truyền user_name vào template
    }
    # --- KẾT THÚC SỬA ĐỔI ---
    return render(request, 'churn_predict/welcome.html', context)


# --- CÁC VIEW ML CỦA BẠN GIỮ NGUYÊN ---

def analysis_view(request):
    context = {'page_title': 'Phân tích dữ liệu'}
    return render(request, 'churn_predict/analysis.html', context)

def explanation_view(request):

    # --- SỬA ĐỔI ---
    # 1. Tạo một chuỗi "cache buster" dựa trên thời gian hiện tại
    # (Để ép trình duyệt luôn tải file mới nhất)
    cache_buster = f"?v={int(time.time())}"

    # 2. Xây dựng đường dẫn URL đầy đủ trong view
    shap_url = static('explanations/shap/shap_summary_GradientBoosting.png') + cache_buster
    lime_url = static('explanations/lime/lime_explanation_GradientBoosting.html') + cache_buster
    # --- KẾT THÚC SỬA ĐỔI ---

    context = {
        'page_title': 'Giải thích mô hình',

        # 3. Gửi URL đã xử lý ra template
        'shap_image_url': shap_url,
        'lime_html_url': lime_url,

        'explanation_title': 'Giải thích cho: Gradient Boosting (Model tốt nhất)',
        'metrics': MODEL_METRICS['gradient_boosting'] 
    }
    return render(request, 'churn_predict/explanation.html', context)
MODEL_METRICS = {
    'logistic_regression': {
        "Accuracy": "73.81%",
        "AUC-ROC": "0.8417",
        "Precision (Churn)": "0.5043",
        "Recall (Churn)": "0.7834",
        "F1-score (Churn)": "0.6136",
    },
    'random_forest': {
        "Accuracy": "78.28%",
        "AUC-ROC": "0.8232",
        "Precision (Churn)": "0.6189",
        "Recall (Churn)": "0.4733",
        "F1-score (Churn)": "0.5364",
    },
    'gradient_boosting': {
        "Accuracy": "74.52%",
        "AUC-ROC": "0.8328",
        "Precision (Churn)": "0.5132",
        "Recall (Churn)": "0.7807",
        "F1-score (Churn)": "0.6193",
    },
}

def predict_view(request):
    context = {
        'page_title': 'Dự đoán',
        'prediction': None,
        'probability': None,
        'form_data': None,
        'show_explain_button': False # MỚI: Mặc định ẩn nút Explain
    }

    if request.method == 'POST':
        try:
            model_path = GB_MODEL_PATH 
            if not os.path.exists(model_path):
                 raise FileNotFoundError(f"Model file not found at {model_path}")
            with open(model_path, 'rb') as f:
                pipeline = pickle.load(f)

            data = {
                'gender': request.POST.get('gender'),
                'SeniorCitizen': int(request.POST.get('SeniorCitizen', 0)),
                'Partner': request.POST.get('Partner'),
                'Dependents': request.POST.get('Dependents'),
                'tenure': int(request.POST.get('tenure', 0)),
                'PhoneService': request.POST.get('PhoneService'),
                'MultipleLines': request.POST.get('MultipleLines'),
                'InternetService': request.POST.get('InternetService'),
                'OnlineSecurity': request.POST.get('OnlineSecurity', 'No'),
                'OnlineBackup': request.POST.get('OnlineBackup', 'No'),
                'DeviceProtection': request.POST.get('DeviceProtection', 'No'),
                'TechSupport': request.POST.get('TechSupport', 'No'),
                'StreamingTV': request.POST.get('StreamingTV', 'No'),
                'StreamingMovies': request.POST.get('StreamingMovies', 'No'),
                'Contract': request.POST.get('Contract'),
                'PaperlessBilling': request.POST.get('PaperlessBilling'),
                'PaymentMethod': request.POST.get('PaymentMethod'),
                'MonthlyCharges': float(request.POST.get('MonthlyCharges', 0.0)),
                'TotalCharges': float(request.POST.get('TotalCharges', 0.0)),
            }

            input_df = pd.DataFrame([data])
            prediction_val = pipeline.predict(input_df)[0]
            prediction_proba_val = pipeline.predict_proba(input_df)[0][1]

            if prediction_val == 1:
                context['prediction'] = "Khách hàng sẽ Churn"
                context['prediction_class'] = "churn-yes"
            else:
                context['prediction'] = "Khách hàng sẽ ở lại"
                context['prediction_class'] = "churn-no"
                
            context['probability'] = f"{prediction_proba_val:.2%}"
            context['form_data'] = data 
            
            # MỚI: Lưu dữ liệu vào session và cho phép hiện nút
            request.session['last_prediction_data'] = data
            context['show_explain_button'] = True

        except Exception as e:
            print(f"Lỗi khi dự đoán: {e}")
            context['prediction'] = f"Lỗi: {e}"
            context['prediction_class'] = "churn-yes"
    
    return render(request, 'churn_predict/predict.html', context)


# --- MỚI: Thêm các View xử lý API Đăng ký, Đăng nhập, Đăng xuất ---

@csrf_exempt # Tắt kiểm tra CSRF cho view API này
def register_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            email = data.get('email')
            password = data.get('password') # Mật khẩu thô
            last_name = data.get('last_name')
            first_name = data.get('first_name')
            phone = data.get('phone')

            # 1. Kiểm tra email đã tồn tại chưa
            if User.objects.filter(email=email).exists():
                return JsonResponse({'success': False, 'error': 'Email đã tồn tại.'})
            
            # 2. Tạo User mới và lưu (lưu mật khẩu thô)
            new_user = User(
                last_name=last_name,
                first_middle_name=first_name,
                email=email,
                phone_number=phone,
                password=password # Lưu mật khẩu thô
            )
            new_user.save()

            # Trả về thành công
            return JsonResponse({'success': True})
        except Exception as e:
            print(f"Lỗi khi đăng ký: {e}") 
            return JsonResponse({'success': False, 'error': 'Có lỗi máy chủ.'})
    return JsonResponse({'success': False, 'error': 'Yêu cầu không hợp lệ.'})


@csrf_exempt # Tắt kiểm tra CSRF cho view API này
def login_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        email = data.get('email')
        password_from_form = data.get('password') # Mật khẩu thô từ form
        
        try:
            # 1. Tìm user bằng email
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            # 2. Nếu không tìm thấy user
            return JsonResponse({'success': False, 'error': 'Email hoặc mật khẩu không chính xác.'})
        
        # 3. Lấy mật khẩu thô từ CSDL
        password_from_db = user.password
        
        # 4. So sánh mật khẩu thô trực tiếp
        if password_from_form == password_from_db:
            # Mật khẩu khớp!
            # Lưu user_id và email vào session để ghi nhớ đăng nhập
            request.session['user_id'] = user.ID # Dùng 'ID' (viết hoa) như trong model
            request.session['user_email'] = user.email
            full_name = f"{user.first_middle_name} {user.last_name}"
            request.session['user_full_name'] = full_name
            
            # Trả về thành công và đường dẫn để JS chuyển hướng
            return JsonResponse({'success': True, 'redirect_url': '/welcome/'}) # Chuyển đến trang welcome
        else:
            # Mật khẩu không khớp
            return JsonResponse({'success': False, 'error': 'Email hoặc mật khẩu không chính xác.'})
    
    return JsonResponse({'success': False, 'error': 'Yêu cầu không hợp lệ.'})


def logout_view(request):
    try:
        # Xóa thông tin session
        del request.session['user_id']
        del request.session['user_email']
    except KeyError:
        pass # Nếu session không có thì bỏ qua
    
    # Quay về trang chủ (trang đăng nhập)
    return redirect('core:index_view') # Sẽ đặt tên URL là 'index_view'

# --- KẾT THÚC MÀN HÌNH MỚI ---
@csrf_exempt # Tạm thời tắt CSRF, vì chúng ta chỉ đọc dữ liệu (GET)
def get_dataset_view(request):
    dataset_key = request.GET.get('dataset', None) # Lấy giá trị từ dropdown
    
    data = []
    headers = []
    
    try:
        if dataset_key == 'gender':
            queryset = Customers.objects.all().values()
            headers = ['customer_id', 'gender', 'is_senior', 'account_tenure_months']
            data = list(queryset)
            
        elif dataset_key == 'churn':
            queryset = ChurnRecords.objects.all().values()
            headers = ['customer_id', 'is_churned', 'total_transaction_value', 'churn_id']
            data = list(queryset)

        elif dataset_key == 'contract':
            queryset = Contracts.objects.all().values()
            headers = ['customer_id', 'contract_type', 'payment_method', 'contract_id']
            data = list(queryset)

        elif dataset_key == 'internet_service':
            queryset = InternetServices.objects.all().values()
            headers = ['customer_id', 'service_type', 'internet_id']
            data = list(queryset)

        elif dataset_key == 'payment_method':
            queryset = PaymentMethods.objects.all().values()
            headers = ['method_name', 'monthly_fee', 'auto_payment', 'e_statement', 'payment_id']
            data = list(queryset)

        elif dataset_key == 'phone_service':
            queryset = PhoneServices.objects.all().values()
            headers = ['customer_id', 'has_multiple_lines', 'phone_id']
            data = list(queryset)

        else:
            return JsonResponse({'error': 'Invalid dataset key'}, status=400)

        # Trả về dữ liệu dưới dạng JSON
        return JsonResponse({'headers': headers, 'rows': data})

    except Exception as e:
        print(f"Lỗi khi truy vấn CSDL: {e}")
        return JsonResponse({'error': str(e)}, status=500)
    
@csrf_exempt
def forgot_password_verify_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            email = data.get('email')
            phone = data.get('phone')

            # 1. Kiểm tra xem email VÀ SĐT có khớp với user nào không
            user = User.objects.get(email=email, phone_number=phone)
            
            # 2. Nếu tìm thấy, lưu email này vào session để dùng ở bước sau
            request.session['password_reset_email'] = user.email
            return JsonResponse({'success': True})

        except User.DoesNotExist:
            # 3. Nếu không tìm thấy
            return JsonResponse({'success': False, 'error': 'Email hoặc số điện thoại không khớp.'})
        except Exception as e:
            print(f"Lỗi khi xác minh quên mật khẩu: {e}") 
            return JsonResponse({'success': False, 'error': 'Có lỗi máy chủ.'})
            
    return JsonResponse({'success': False, 'error': 'Yêu cầu không hợp lệ.'})


# --- MỚI: API CẬP NHẬT MẬT KHẨU MỚI ---
@csrf_exempt
def set_new_password_view(request):
    if request.method == 'POST':
        try:
            # 1. Kiểm tra xem user đã được xác minh ở bước 1 chưa (qua session)
            email_to_reset = request.session.get('password_reset_email')
            
            if not email_to_reset:
                return JsonResponse({'success': False, 'error': 'Phiên làm việc hết hạn. Vui lòng thử lại từ đầu.'})

            data = json.loads(request.body)
            new_password = data.get('new_password') # Lấy mật khẩu thô

            # 2. Tìm user và cập nhật mật khẩu (lưu thô)
            user = User.objects.get(email=email_to_reset)
            user.password = new_password # Cập nhật mật khẩu thô
            user.save() # Lưu vào CSDL

            # 3. Xóa session sau khi hoàn tất
            del request.session['password_reset_email']
            
            return JsonResponse({'success': True})
            
        except User.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Không tìm thấy người dùng để cập nhật.'})
        except Exception as e:
            print(f"Lỗi khi đặt mật khẩu mới: {e}") 
            return JsonResponse({'success': False, 'error': 'Có lỗi máy chủ.'})
            
    return JsonResponse({'success': False, 'error': 'Yêu cầu không hợp lệ.'})
# --- HÀM HELPER MỚI: LẤY TỪ PIPELINE (Cần cho LIME) ---
def load_and_clean_for_lime(path):
    correct_path = os.path.join(settings.BASE_DIR.parent, 'dataset', 'Telco-Customer-Churn.csv')

    try:
        df = pd.read_csv(correct_path) # Đọc từ đường dẫn chính xác
    except FileNotFoundError:
        print(f"Lỗi LIME: Không tìm thấy file dataset tại {correct_path}")
        return None, None
        
    df.columns = [c.strip() for c in df.columns]
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    if 'Churn' in df.columns:
        df.rename(columns={'Churn': 'churn'}, inplace=True)
        df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})
    else:
        # Nếu file upload không có cột Churn, không sao, LIME không cần 'y'
        pass
        
    if 'churn' in df.columns:
        X = df.drop(columns=['churn'])
        return X, df['churn']
    else:
        return df, None # Trả về X (df) và y (None)

# --- VIEW API MỚI: CHO NÚT "EXPLAIN" (LIME) ---
# do_an/churn_predict/core/views.py

@csrf_exempt
def explain_lime_view(request):
    if 'user_id' not in request.session:
        return JsonResponse({'success': False, 'error': 'Đã hết phiên làm việc. Vui lòng đăng nhập lại.'})

    if LimeTabularExplainer is None:
        return JsonResponse({'success': False, 'error': 'Thư viện LIME chưa được cài đặt trên server.'})

    try:
        # 1. Lấy dữ liệu dự đoán đã lưu
        data_dict = request.session.get('last_prediction_data')
        if not data_dict:
            return JsonResponse({'success': False, 'error': 'Không tìm thấy dữ liệu dự đoán. Vui lòng nhấn "Predict" trước.'})

        # 2. Load model
        model_path = GB_MODEL_PATH
        with open(model_path, 'rb') as f:
            pipeline = pickle.load(f)

        # 3. Load dữ liệu nền (background data)
        X, y = load_and_clean_for_lime(None) # Hàm này đã tự biết đường dẫn
        if X is None:
            return JsonResponse({'success': False, 'error': 'Không tìm thấy file dataset gốc trên server để chạy LIME.'})

        # 4. Lấy các thành phần từ pipeline
        preprocessor = pipeline.named_steps['preprocessor']
        classifier = pipeline.named_steps['classifier']
        feature_names = preprocessor.get_feature_names_out()
        categorical_features = [col for col in X.columns if X[col].dtype == 'object']
        categorical_indices = [list(X.columns).index(col) for col in categorical_features]

        # 5. Khởi tạo LIME
        explainer = LimeTabularExplainer(
            training_data=preprocessor.transform(X),
            feature_names=feature_names,
            class_names=['No Churn', 'Churn'],
            categorical_features=categorical_indices,
            mode='classification',
            random_state=42
        )

        # 6. Biến đổi dữ liệu nhập vào
        instance_df = pd.DataFrame([data_dict])
        instance_transformed = preprocessor.transform(instance_df)

        # 7. Chạy giải thích (CHẬM)
        predict_fn = lambda x: classifier.predict_proba(x)
        exp = explainer.explain_instance(
            data_row=instance_transformed[0], 
            predict_fn=predict_fn, 
            num_features=10
        )

        # --- SỬA ĐỔI LỚN BẮT ĐẦU TỪ ĐÂY ---

        # 8. Lưu kết quả LIME ra file HTML tạm
        html_content = exp.as_html()

        # Tạo một tên file tạm thời, dùng ID user để tránh xung đột
        user_id = request.session.get('user_id', 'temp')
        filename = f'lime_explain_{user_id}.html'

        # Đường dẫn để LƯU file (file system path)
        save_dir = settings.BASE_DIR / "static" / "explanations" / "lime"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # 9. Trả về ĐƯỜNG DẪN (URL) của file đó
        # Thêm "cache-buster" (dấu ?v=...) để đảm bảo trình duyệt
        # luôn tải file mới nhất, không dùng file cũ
        cache_buster = f"?v={int(time.time())}"
        file_url = static(f'explanations/lime/{filename}') + cache_buster

        return JsonResponse({'success': True, 'url': file_url})
        # --- KẾT THÚC SỬA ĐỔI ---

    except Exception as e:
        print(f"Lỗi khi chạy LIME: {e}")
        return JsonResponse({'success': False, 'error': f'Lỗi server khi chạy LIME: {e}'})
    
# --- VIEW API MỚI: CHO "UPLOAD & RETRAIN" ---
def upload_retrain_view(request):
    if 'user_id' not in request.session:
        return redirect('core:index_view')
        
    if request.method == 'POST':
        csv_file = request.FILES.get('new_dataset')
        
        if not csv_file:
            messages.error(request, 'Bạn chưa chọn file.')
            return redirect('core:welcome')
            
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'Lỗi: File phải có định dạng .csv')
            return redirect('core:welcome')

        try:
            # 1. Lưu file .csv mới, ghi đè file cũ
            save_path = os.path.join(settings.BASE_DIR, 'dataset', 'Telco-Customer-Churn.csv')
            
            # Xóa file cũ nếu tồn tại
            if default_storage.exists(save_path):
                default_storage.delete(save_path)
                
            # Lưu file mới
            default_storage.save(save_path, csv_file)

            # 2. Kích hoạt kịch bản churn_pipeline.py chạy ngầm
            
            # Đường dẫn đến file python trong môi trường ảo (venv)
            python_exe = sys.executable 
            # Đường dẫn đến kịch bản pipeline
            script_path = os.path.join(settings.BASE_DIR, 'ml_models', 'churn_pipeline.py')
            # Thư mục làm việc (để script biết `../output_models` ở đâu)
            working_dir = os.path.join(settings.BASE_DIR, 'ml_models')
            
            # Chạy tiến trình mới trong nền (không đợi nó)
            subprocess.Popen([python_exe, script_path], cwd=working_dir)
            
            messages.success(request, (
                'File đã được tải lên! Quá trình huấn luyện lại model đã bắt đầu trong nền. '
                'Vui lòng đợi 3-5 phút trước khi sử dụng các model mới.'
            ))
            
        except Exception as e:
            print(f"Lỗi khi upload: {e}")
            messages.error(request, f'Lỗi khi xử lý file: {e}')
            
        return redirect('core:welcome')
        
    return redirect('core:welcome')