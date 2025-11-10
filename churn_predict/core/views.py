# do_an/churn_predict/core/views.py

import pickle
import os
import pandas as pd
from django.shortcuts import render, redirect # --- SỬA ĐỔI --- (Thêm 'redirect')
from django.conf import settings

# --- MỚI: Thêm các thư viện cho logic Auth ---
from django.http import JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt # Sẽ dùng cho API
from .models import User # Import Model 'User' chúng ta đã tạo
from .models import (
    User, Contracts, Customers, InternetServices, 
    PaymentMethods, PhoneServices, ChurnRecords
)
# --- KẾT THÚC MỚI ---


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
    context = {
        'page_title': 'Giải thích mô hình',
        'shap_image_url': None,     # Đường dẫn ảnh SHAP
        'lime_html_url': None,      # Đường dẫn file LIME
        'form_data': None,          # Để giữ lại lựa chọn dropdown
        'explanation_title': None,  # Tiêu đề cho kết quả
    }

    if request.method == 'POST':
        # Lấy lựa chọn thuật toán từ form
        algo = request.POST.get('algorithm')
        
        # Lưu lựa chọn của form để hiển thị lại
        context['form_data'] = request.POST 

        if algo == 'logistic_regression':
            context['shap_image_url'] = 'explanations/shap/shap_summary_LogisticRegression.png'
            context['lime_html_url'] = 'explanations/lime/lime_explanation_LogisticRegression.html'
            context['explanation_title'] = 'Giải thích cho: Logistic Regression'
        
        elif algo == 'random_forest':
            context['shap_image_url'] = 'explanations/shap/shap_summary_RandomForest.png'
            context['lime_html_url'] = 'explanations/lime/lime_explanation_RandomForest.html'
            context['explanation_title'] = 'Giải thích cho: Random Forest'

        elif algo == 'gradient_boosting': # Giá trị này phải khớp với <option>
            context['shap_image_url'] = 'explanations/shap/shap_summary_GradientBoosting.png'
            context['lime_html_url'] = 'explanations/lime/lime_explanation_GradientBoosting.html'
            context['explanation_title'] = 'Giải thích cho: Gradient Boosting'

    # Render lại trang với context đã chứa (hoặc không chứa) đường dẫn
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

# --- SỬA ĐỔI HOÀN TOÀN 'predict_view' ---
def predict_view(request):
    context = {
        'page_title': 'Dự đoán',
        'prediction': None,
        'probability': None,
        'metrics': None,
        'form_data': None # Sẽ dùng để giữ lại giá trị form
    }

    if request.method == 'POST':
        try:
            # --- 1. Load Model ---
            algo_choice = request.POST.get('algorithm')
            model_path = ""
            
            if algo_choice == 'logistic_regression':
                model_path = LR_MODEL_PATH
                context['metrics'] = MODEL_METRICS['logistic_regression']
            elif algo_choice == 'random_forest':
                model_path = RF_MODEL_PATH
                context['metrics'] = MODEL_METRICS['random_forest']
            elif algo_choice == 'gradient_boosting': # Đổi giá trị này trong HTML
                model_path = GB_MODEL_PATH
                context['metrics'] = MODEL_METRICS['gradient_boosting']
            else:
                # Nếu không chọn, mặc định là Random Forest
                model_path = RF_MODEL_PATH
                context['metrics'] = MODEL_METRICS['random_forest']
            
            if not os.path.exists(model_path):
                 raise FileNotFoundError(f"Model file not found at {model_path}")

            with open(model_path, 'rb') as f:
                pipeline = pickle.load(f)

            # --- 2. Thu thập dữ liệu từ Form ---
            # Lưu ý: 'name' trong HTML phải khớp chính xác
            data = {
                # Identity
                'gender': request.POST.get('gender'),
                'SeniorCitizen': int(request.POST.get('SeniorCitizen', 0)),
                'Partner': request.POST.get('Partner'),
                'Dependents': request.POST.get('Dependents'),
                'tenure': int(request.POST.get('tenure', 0)),
                
                # Services
                'PhoneService': request.POST.get('PhoneService'),
                'MultipleLines': request.POST.get('MultipleLines'),
                'InternetService': request.POST.get('InternetService'),
                
                # Internet Sub-services (Checkboxes)
                # Nếu không check, POST không gửi gì. .get(name, 'No') sẽ đổi thành 'No'
                'OnlineSecurity': request.POST.get('OnlineSecurity', 'No'),
                'OnlineBackup': request.POST.get('OnlineBackup', 'No'),
                'DeviceProtection': request.POST.get('DeviceProtection', 'No'),
                'TechSupport': request.POST.get('TechSupport', 'No'),
                'StreamingTV': request.POST.get('StreamingTV', 'No'),
                'StreamingMovies': request.POST.get('StreamingMovies', 'No'),
                
                # Contract
                'Contract': request.POST.get('Contract'),
                'PaperlessBilling': request.POST.get('PaperlessBilling'),
                'PaymentMethod': request.POST.get('PaymentMethod'),
                'MonthlyCharges': float(request.POST.get('MonthlyCharges', 0.0)),
                'TotalCharges': float(request.POST.get('TotalCharges', 0.0)),
            }

            # --- 3. Tạo DataFrame và Dự đoán ---
            input_df = pd.DataFrame([data])
            
            prediction_val = pipeline.predict(input_df)[0]
            prediction_proba_val = pipeline.predict_proba(input_df)[0][1] # Xác suất Churn (lớp 1)

            # --- 4. Gửi kết quả về context ---
            if prediction_val == 1:
                context['prediction'] = "Khách hàng sẽ Churn"
                context['prediction_class'] = "churn-yes" # Dùng cho CSS
            else:
                context['prediction'] = "Khách hàng sẽ ở lại"
                context['prediction_class'] = "churn-no" # Dùng cho CSS
                
            context['probability'] = f"{prediction_proba_val:.2%}" # Format %
            context['form_data'] = data # Gửi lại dữ liệu form để giữ giá trị

        except Exception as e:
            print(f"Lỗi khi dự đoán: {e}")
            context['prediction'] = f"Lỗi: {e}" # Hiển thị lỗi ra giao diện
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