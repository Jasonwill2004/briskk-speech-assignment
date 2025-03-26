# 🎯 Speech-to-Text Search Implementation

## 🌟 Overview
A real-time speech recognition and search suggestion system built with FastAPI, OpenAI Whisper, and WebSocket support. Deployed on AWS ECS for scalability and reliability.

## 🚀 Key Features
- Real-time speech recognition using OpenAI Whisper
- WebSocket support for continuous audio streaming
- Smart search suggestions with AI-based ranking
- Noise-resilient audio processing
- Containerized deployment on AWS ECS
- Auto-scaling and high availability

## 💡 Technical Choices & Trade-offs

### 1. Speech Recognition: Whisper vs DeepSpeech
- **Chose Whisper because:**
  - Better accuracy on noisy inputs
  - Smaller model size (tiny model: 39M parameters)
  - Faster inference time
  - Multi-language support out of the box
- **Trade-offs:**
  - DeepSpeech offers better offline support
  - Whisper requires more RAM (mitigated by using tiny model)

### 2. Data Storage: In-Memory vs Redis
- **Chose In-Memory Storage because:**
  - Simpler deployment architecture
  - Sufficient for demonstration purposes
  - Lower latency for small datasets
- **Trade-offs:**
  - Redis would be better for production scale
  - Missing persistence across container restarts

### 3. Deployment: AWS ECS vs Lambda
- **Chose ECS because:**
  - WebSocket support required
  - Better for long-running connections
  - More cost-effective for continuous workloads
- **Trade-offs:**
  - Lambda would be cheaper for sporadic usage
  - ECS requires more configuration

## 📊 Performance Metrics
- Speech recognition accuracy: 95%
- Average response time: <500ms
- WebSocket latency: ~100ms
- Memory usage: ~800MB

## 🎯 Task Completion Screenshots

### Task 1: Speech Recognition API
<img width="1466" alt="Screenshot 2025-03-26 at 10 08 54 AM" src="https://github.com/user-attachments/assets/0bf71a2c-1642-4735-8513-53d553b58916" /> 


- Implemented FastAPI endpoint
- Achieved 95% accuracy on clean audio
- Response time under 500ms

### Task 2: Noisy Audio Handling
<img width="1470" alt="Screenshot 2025-03-26 at 10 11 08 AM" src="https://github.com/user-attachments/assets/6b0e1b01-3665-460b-b333-a7d87f7f63c2" />


- Implemented noise reduction
- Improved accuracy from 75% to 92% on noisy audio
- Processing time: 800ms

### Task3: Smart Search Autocomplete
<img width="1470" alt="Screenshot 2025-03-26 at 10 12 10 AM" src="https://github.com/user-attachments/assets/253b5c88-2aee-4c4d-afa2-6c031e18a7b1" />
 


- Implemented AI-based ranking
- Response time: 200ms
- Top suggestions match user intent

### Task 4: WebSocket Implementation
<img width="666" alt="Screenshot 2025-03-26 at 10 20 33 AM" src="https://github.com/user-attachments/assets/7662f09c-59dd-4007-acf8-4eb26d2f3897" />


- Real-time audio streaming
- Continuous transcription
- Dynamic suggestions


## Video Explanation


## 🛠️ API Endpoints
```bash
# REST Endpoints
POST /api/voice-to-text
GET /api/autocomplete?q={query}

# WebSocket Endpoint
ws://speech-search-alb-607098999.eu-north-1.elb.amazonaws.com:8000/ws/speech-to-search
```

## 📦 Deployment
- Region: eu-north-1 (Stockholm)
- Container Registry: Amazon ECR
- Compute: AWS ECS Fargate
- Load Balancer: Application Load Balancer

## 🧪 Testing Instructions
```bash
# Health check
curl http://speech-search-alb-607098999.eu-north-1.elb.amazonaws.com:8000/health

# WebSocket test
wscat -c ws://speech-search-alb-607098999.eu-north-1.elb.amazonaws.com:8000/ws/speech-to-search
```

## 📈 Future Improvements
1. Implement Redis for persistent storage
2. Add user authentication
3. Implement SSL/TLS for secure WebSocket
4. Add custom domain and CDN
5. Implement rate limiting

## 🔗 Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Whisper Documentation](https://github.com/openai/whisper)
- [AWS ECS Best Practices](https://docs.aws.amazon.com/AmazonECS/latest/bestpracticesguide/ecs-bp.html)
