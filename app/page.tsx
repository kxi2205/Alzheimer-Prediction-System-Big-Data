import PatientAssessmentForm from '@/components/PatientAssessmentForm'
import { Toaster } from 'react-hot-toast'

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-100 py-8">
      <div className="container mx-auto px-4">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Alzheimer's Disease Risk Assessment
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Complete this comprehensive assessment to evaluate your risk factors for Alzheimer's disease. 
            This tool uses advanced machine learning models trained on clinical data to provide personalized risk insights.
          </p>
        </div>
        
        <PatientAssessmentForm />
        
        <div className="mt-12 text-center text-sm text-gray-500">
          <p>
            This assessment is for educational and informational purposes only. 
            Always consult with healthcare professionals for medical advice.
          </p>
        </div>
      </div>
      
      <Toaster 
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
          },
        }}
      />
    </div>
  )
}
