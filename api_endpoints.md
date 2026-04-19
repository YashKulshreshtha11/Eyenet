# API Endpoints Documentation - EyeNet Project

## 1. API Structure Overview

### 1.1 API Architecture Diagram

```
+----------------+     +----------------+     +----------------+     +----------------+
|                |     |                |     |                |     |                |
|   Frontend     |---->|   API Gateway  |---->|   FastAPI      |---->|   Services     |
|   (React)      |     |   (Auth/Rate)   |     |   Router       |     |   Layer        |
|                |     |                |     |                |     |                |
+----------------+     +----------------+     +----------------+     +----------------+
         |                      |                      |                      |
         |                      |                      |                      |
         v                      v                      v                      v
+----------------+     +----------------+     +----------------+     +----------------+
|                |     |                |     |                |     |                |
|   HTTP/HTTPS   |---->|   Middleware    |---->|   Route        |---->|   Business     |
|   Requests     |     |   (Validation)  |     |   Handlers     |     |   Logic        |
|                |     |                |     |                |     |                |
+----------------+     +----------------+     +----------------+     +----------------+
         |                      |                      |                      |
         |                      |                      |                      |
         v                      v                      v                      v
+----------------+     +----------------+     +----------------+     +----------------+
|                |     |                |     |                |     |                |
|   JSON/XML     |---->|   Request/      |---->|   Database     |---->|   Response     |
|   Responses    |     |   Response      |     |   Operations    |     |   Formatting   |
|                |     |   Processing    |     |                |     |                |
+----------------+     +----------------+     +----------------+     +----------------+
```

### 1.2 API Endpoint Categories

```
/api/v1/
  |- Authentication Endpoints
  |   |- POST /login
  |   |- POST /logout
  |   |- POST /refresh
  |
  |- Image Management Endpoints
  |   |- POST /upload
  |   |- GET /images/{image_id}
  |   |- DELETE /images/{image_id}
  |
  |- Prediction Endpoints
  |   |- POST /predict
  |   |- GET /predict/{prediction_id}
  |   |- GET /predict/history/{user_id}
  |
  |- Report Endpoints
  |   |- GET /report/{prediction_id}
  |   |- POST /report/generate
  |   |- GET /report/download/{report_id}
  |
  |- Model Management Endpoints
  |   |- GET /models
  |   |- GET /models/{model_id}
  |   |- POST /models/evaluate
  |
  |- User Management Endpoints
  |   |- GET /users/{user_id}
  |   |- PUT /users/{user_id}
  |   |- GET /users/{user_id}/history
  |
  |- System Endpoints
  |   |- GET /health
  |   |- GET /metrics
  |   |- GET /version
```

## 2. Detailed API Endpoints

### 2.1 Authentication Endpoints

#### 2.1.1 User Login
```
POST /api/v1/login
Content-Type: application/json

Request Body:
{
  "email": "doctor@hospital.com",
  "password": "secure_password"
}

Response (200 OK):
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "user_123",
    "email": "doctor@hospital.com",
    "role": "doctor",
    "name": "Dr. John Smith"
  }
}

Response (401 Unauthorized):
{
  "error": "Invalid credentials",
  "message": "Email or password is incorrect"
}
```

#### 2.1.2 User Logout
```
POST /api/v1/logout
Authorization: Bearer {access_token}

Response (200 OK):
{
  "message": "Successfully logged out"
}
```

#### 2.1.3 Token Refresh
```
POST /api/v1/refresh
Content-Type: application/json

Request Body:
{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}

Response (200 OK):
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### 2.2 Image Management Endpoints

#### 2.2.1 Upload Image
```
POST /api/v1/upload
Authorization: Bearer {access_token}
Content-Type: multipart/form-data

Request Body:
- file: [image file] (JPEG/PNG, max 10MB)
- patient_id: "patient_456" (optional)
- metadata: {
    "age": 45,
    "gender": "male",
    "notes": "Routine checkup"
  } (optional)

Response (201 Created):
{
  "image_id": "img_789",
  "filename": "retinal_image.jpg",
  "size": 2048576,
  "format": "JPEG",
  "upload_time": "2024-01-15T10:30:00Z",
  "status": "uploaded",
  "patient_id": "patient_456",
  "metadata": {
    "age": 45,
    "gender": "male",
    "notes": "Routine checkup"
  }
}

Response (400 Bad Request):
{
  "error": "Invalid file format",
  "message": "Only JPEG and PNG images are supported"
}
```

#### 2.2.2 Get Image Details
```
GET /api/v1/images/{image_id}
Authorization: Bearer {access_token}

Response (200 OK):
{
  "image_id": "img_789",
  "filename": "retinal_image.jpg",
  "size": 2048576,
  "format": "JPEG",
  "upload_time": "2024-01-15T10:30:00Z",
  "patient_id": "patient_456",
  "metadata": {
    "age": 45,
    "gender": "male",
    "notes": "Routine checkup"
  },
  "predictions": [
    {
      "prediction_id": "pred_123",
      "timestamp": "2024-01-15T10:31:00Z",
      "model_version": "v2.0.0"
    }
  ]
}
```

#### 2.2.3 Delete Image
```
DELETE /api/v1/images/{image_id}
Authorization: Bearer {access_token}

Response (200 OK):
{
  "message": "Image deleted successfully"
}
```

### 2.3 Prediction Endpoints

#### 2.3.1 Create Prediction
```
POST /api/v1/predict
Authorization: Bearer {access_token}
Content-Type: application/json

Request Body:
{
  "image_id": "img_789",
  "model_version": "v2.0.0" (optional, defaults to latest)
}

Response (202 Accepted):
{
  "prediction_id": "pred_123",
  "image_id": "img_789",
  "status": "processing",
  "estimated_time": 30,
  "created_at": "2024-01-15T10:31:00Z",
  "model_version": "v2.0.0"
}

Response (400 Bad Request):
{
  "error": "Image not found",
  "message": "The specified image ID does not exist"
}
```

#### 2.3.2 Get Prediction Results
```
GET /api/v1/predict/{prediction_id}
Authorization: Bearer {access_token}

Response (200 OK):
{
  "prediction_id": "pred_123",
  "image_id": "img_789",
  "status": "completed",
  "created_at": "2024-01-15T10:31:00Z",
  "completed_at": "2024-01-15T10:31:30Z",
  "model_version": "v2.0.0",
  "results": {
    "predictions": [
      {
        "class": "Normal",
        "probability": 0.85,
        "confidence": "high"
      },
      {
        "class": "Diabetic Retinopathy",
        "probability": 0.10,
        "confidence": "low"
      },
      {
        "class": "Cataract",
        "probability": 0.03,
        "confidence": "low"
      },
      {
        "class": "Glaucoma",
        "probability": 0.02,
        "confidence": "low"
      }
    ],
    "primary_prediction": {
      "class": "Normal",
      "probability": 0.85,
      "confidence": "high"
    },
    "explanation": {
      "grad_cam_available": true,
      "heatmap_url": "/api/v1/explanation/heatmap/pred_123",
      "feature_importance": {
        "optic_disc": 0.45,
        "blood_vessels": 0.30,
        "macula": 0.15,
        "retina": 0.10
      }
    },
    "processing_time": 28.5,
    "model_confidence": 0.92
  }
}

Response (202 Accepted):
{
  "prediction_id": "pred_123",
  "status": "processing",
  "estimated_time": 15,
  "progress": 65
}
```

#### 2.3.3 Get User Prediction History
```
GET /api/v1/predict/history/{user_id}
Authorization: Bearer {access_token}
Query Parameters:
- limit: 20 (default)
- offset: 0 (default)
- start_date: 2024-01-01 (optional)
- end_date: 2024-01-31 (optional)

Response (200 OK):
{
  "total_predictions": 45,
  "predictions": [
    {
      "prediction_id": "pred_123",
      "image_id": "img_789",
      "timestamp": "2024-01-15T10:31:00Z",
      "primary_prediction": "Normal",
      "confidence": 0.85,
      "model_version": "v2.0.0"
    }
  ],
  "pagination": {
    "limit": 20,
    "offset": 0,
    "total": 45
  }
}
```

### 2.4 Report Endpoints

#### 2.4.1 Generate Report
```
POST /api/v1/report/generate
Authorization: Bearer {access_token}
Content-Type: application/json

Request Body:
{
  "prediction_id": "pred_123",
  "report_type": "detailed", (options: "summary", "detailed", "medical")
  "include_heatmap": true,
  "patient_info": {
    "name": "John Doe",
    "age": 45,
    "gender": "male",
    "patient_id": "patient_456"
  }
}

Response (201 Created):
{
  "report_id": "report_456",
  "prediction_id": "pred_123",
  "status": "generating",
  "estimated_time": 10,
  "created_at": "2024-01-15T10:32:00Z"
}
```

#### 2.4.2 Get Report
```
GET /api/v1/report/{report_id}
Authorization: Bearer {access_token}

Response (200 OK):
{
  "report_id": "report_456",
  "prediction_id": "pred_123",
  "status": "completed",
  "report_type": "detailed",
  "generated_at": "2024-01-15T10:32:10Z",
  "content": {
    "summary": "The retinal image shows normal characteristics...",
    "findings": [
      "Optic disc appears normal",
      "No signs of diabetic retinopathy",
      "Clear lens, no cataract detected",
      "Normal intraocular pressure indicators"
    ],
    "recommendations": [
      "Routine follow-up in 12 months",
      "Maintain regular eye examinations"
    ],
    "confidence_score": 0.85
  },
  "download_urls": {
    "pdf": "/api/v1/report/download/report_456?format=pdf",
    "html": "/api/v1/report/download/report_456?format=html"
  }
}
```

#### 2.4.3 Download Report
```
GET /api/v1/report/download/{report_id}
Authorization: Bearer {access_token}
Query Parameters:
- format: "pdf" (default) or "html"

Response (200 OK):
Content-Type: application/pdf
Content-Disposition: attachment; filename="eyenet_report_456.pdf

[PDF file content]
```

### 2.5 Model Management Endpoints

#### 2.5.1 Get Available Models
```
GET /api/v1/models
Authorization: Bearer {access_token}

Response (200 OK):
{
  "models": [
    {
      "model_id": "resnet18_v2",
      "name": "ResNet-18 v2.0",
      "version": "2.0.0",
      "type": "cnn",
      "status": "active",
      "accuracy": 0.94,
      "description": "ResNet-18 trained on ODIR5K dataset"
    },
    {
      "model_id": "efficientnet_v2",
      "name": "EfficientNet-B0 v2.0",
      "version": "2.0.0",
      "type": "cnn",
      "status": "active",
      "accuracy": 0.93,
      "description": "EfficientNet-B0 with compound scaling"
    },
    {
      "model_id": "densenet_v2",
      "name": "DenseNet-121 v2.0",
      "version": "2.0.0",
      "type": "cnn",
      "status": "active",
      "accuracy": 0.92,
      "description": "DenseNet-121 with dense connectivity"
    },
    {
      "model_id": "custom_cnn_v2",
      "name": "Custom CNN v2.0",
      "version": "2.0.0",
      "type": "cnn",
      "status": "active",
      "accuracy": 0.91,
      "description": "Custom CNN optimized for retinal images"
    }
  ]
}
```

#### 2.5.2 Get Model Details
```
GET /api/v1/models/{model_id}
Authorization: Bearer {access_token}

Response (200 OK):
{
  "model_id": "resnet18_v2",
  "name": "ResNet-18 v2.0",
  "version": "2.0.0",
  "type": "cnn",
  "status": "active",
  "accuracy": 0.94,
  "precision": 0.93,
  "recall": 0.95,
  "f1_score": 0.94,
  "training_dataset": "ODIR5K + Eye Diseases Classification",
  "training_date": "2024-01-10T00:00:00Z",
  "parameters": {
    "input_size": [224, 224, 3],
    "num_classes": 4,
    "architecture": "resnet18"
  },
  "performance_metrics": {
    "normal": {
      "precision": 0.96,
      "recall": 0.94,
      "f1_score": 0.95
    },
    "diabetic_retinopathy": {
      "precision": 0.92,
      "recall": 0.94,
      "f1_score": 0.93
    },
    "cataract": {
      "precision": 0.91,
      "recall": 0.93,
      "f1_score": 0.92
    },
    "glaucoma": {
      "precision": 0.93,
      "recall": 0.92,
      "f1_score": 0.92
    }
  }
}
```

### 2.6 System Endpoints

#### 2.6.1 Health Check
```
GET /api/v1/health

Response (200 OK):
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "2.0.0",
  "services": {
    "database": "healthy",
    "ai_models": "healthy",
    "file_storage": "healthy"
  },
  "uptime": 86400
}
```

#### 2.6.2 System Metrics
```
GET /api/v1/metrics
Authorization: Bearer {access_token}

Response (200 OK):
{
  "system": {
    "cpu_usage": 45.2,
    "memory_usage": 67.8,
    "disk_usage": 23.4,
    "gpu_usage": 78.9
  },
  "api": {
    "requests_per_minute": 120,
    "average_response_time": 250,
    "error_rate": 0.02,
    "active_connections": 45
  },
  "models": {
    "predictions_today": 1250,
    "average_confidence": 0.87,
    "processing_time_avg": 28.5
  }
}
```

#### 2.6.3 API Version
```
GET /api/v1/version

Response (200 OK):
{
  "api_version": "2.0.0",
  "model_version": "2.0.0",
  "build_date": "2024-01-10T12:00:00Z",
  "git_commit": "abc123def456",
  "environment": "production"
}
```

## 3. API Request/Response Flow

### 3.1 Prediction Flow Diagram

```
+----------------+     +----------------+     +----------------+     +----------------+
|                |     |                |     |                |     |                |
|   Upload Image |---->|   Validate     |---->|   Queue        |---->|   Process      |
|   Request      |     |   Request      |     |   Prediction   |     |   Image        |
|                |     |                |     |                |     |                |
+----------------+     +----------------+     +----------------+     +----------------+
         |                       |                       |                       |
         |                       |                       |                       |
         v                       v                       v                       v
+----------------+     +----------------+     +----------------+     +----------------+
|                |     |                |     |                |     |                |
|   Store Image  |<-----|   Return       |<-----|   Run Models   |<-----|   Generate     |
|   & Metadata   |     |   Prediction ID |     |   Ensemble     |     |   Features     |
|                |     |                |     |                |     |                |
+----------------+     +----------------+     +----------------+     +----------------+
         |                       |                       |                       |
         |                       |                       |                       |
         v                       v                       v                       v
+----------------+     +----------------+     +----------------+     +----------------+
|                |     |                |     |                |     |                |
|   Notify       |---->|   Store       |---->|   Calculate    |---->|   Apply        |
|   Client       |     |   Results     |     |   Probabilities|     |   Grad-CAM     |
|                |     |                |     |                |     |                |
+----------------+     +----------------+     +----------------+     +----------------+
```

### 3.2 Error Handling Structure

```
API Error Response Format:
{
  "error": "ERROR_CODE",
  "message": "Human readable error message",
  "details": {
    "field": "specific field with error",
    "value": "invalid value provided"
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456"
}

Common Error Codes:
- 400: Bad Request (Invalid input data)
- 401: Unauthorized (Invalid/missing token)
- 403: Forbidden (Insufficient permissions)
- 404: Not Found (Resource doesn't exist)
- 429: Too Many Requests (Rate limit exceeded)
- 500: Internal Server Error (System failure)
- 503: Service Unavailable (System maintenance)
```

## 4. API Security and Authentication

### 4.1 Authentication Flow

```
+----------------+     +----------------+     +----------------+     +----------------+
|                |     |                |     |                |     |                |
|   User         |---->|   Login        |---->|   Validate     |---->|   Generate     |
|   Credentials  |     |   Request      |     |   Credentials  |     |   JWT Token    |
|                |     |                |     |                |     |                |
+----------------+     +----------------+     +----------------+     +----------------+
         |                       |                       |                       |
         |                       |                       |                       |
         v                       v                       v                       v
+----------------+     +----------------+     +----------------+     +----------------+
|                |     |                |     |                |     |                |
|   Return      |<-----|   Store       |<-----|   Verify       |<-----|   Return      |
|   JWT Token    |     |   Token        |     |   Token        |     |   Token       |
|                |     |                |     |                |     |                |
+----------------+     +----------------+     +----------------+     +----------------+
```

### 4.2 Rate Limiting

```
Rate Limits by Endpoint Category:
- Authentication: 5 requests per minute
- Image Upload: 10 requests per minute
- Predictions: 30 requests per minute
- Reports: 20 requests per minute
- System/Metrics: 100 requests per minute

Rate Limit Headers:
X-RateLimit-Limit: 30
X-RateLimit-Remaining: 25
X-RateLimit-Reset: 1642248000
```

This comprehensive API documentation provides a complete reference for all EyeNet system endpoints, ensuring proper integration and usage of the retinal disease detection API.
