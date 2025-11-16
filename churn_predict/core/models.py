# do_an/churn_predict/core/models.py

from django.db import models

# Định nghĩa Model "User" trỏ đến bảng "user" có sẵn
class User(models.Model):
    # Cột ID (viết hoa)
    ID = models.AutoField(primary_key=True, db_column='ID') 
    
    # Các cột còn lại
    last_name = models.CharField(max_length=16)
    first_middle_name = models.CharField(max_length=32)
    email = models.CharField(max_length=45, unique=True)
    phone_number = models.CharField(max_length=45)
    password = models.CharField(max_length=255) # Theo yêu cầu là varchar(255)

    class Meta:
        managed = False  # Quan trọng: Báo Django không quản lý (tạo/xóa/sửa) bảng này
        db_table = 'user' # Quan trọng: Tên chính xác của bảng trong MySQL

    def __str__(self):
        return self.email
    
class Contracts(models.Model):
    # Giả sử 'contract_id' là khóa chính nếu nó là duy nhất,
    # nhưng chúng ta sẽ dùng 'managed = False' nên chỉ cần định nghĩa
    contract_id = models.BigIntegerField(primary_key=True)
    customer_id = models.TextField()
    contract_type = models.TextField()
    payment_method = models.TextField()

    class Meta:
        managed = False
        db_table = 'contracts'


class Customers(models.Model):
    # Giả sử 'customer_id' là khóa chính
    customer_id = models.TextField(primary_key=True)
    gender = models.TextField()
    is_senior = models.BigIntegerField()
    account_tenure_months = models.BigIntegerField()

    class Meta:
        managed = False
        db_table = 'customers'


class InternetServices(models.Model):
    internet_id = models.BigIntegerField(primary_key=True)
    customer_id = models.TextField()
    service_type = models.TextField()

    class Meta:
        managed = False
        db_table = 'internet_services'


class PaymentMethods(models.Model):
    payment_id = models.BigIntegerField(primary_key=True)
    method_name = models.TextField()
    monthly_fee = models.FloatField() # 'double' trong SQL là 'FloatField'
    auto_payment = models.TextField()
    e_statement = models.TextField()

    class Meta:
        managed = False
        db_table = 'payment_methods'


class PhoneServices(models.Model):
    phone_id = models.BigIntegerField(primary_key=True)
    customer_id = models.TextField()
    has_multiple_lines = models.TextField()

    class Meta:
        managed = False
        db_table = 'phone_services'


class ChurnRecords(models.Model):
    churn_id = models.BigIntegerField(primary_key=True)
    customer_id = models.TextField()
    is_churned = models.TextField()
    total_transaction_value = models.TextField()

    class Meta:
        managed = False
        db_table = 'churn_records'
