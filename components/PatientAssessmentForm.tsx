'use client'

import React, { useState, useEffect } from 'react'
import { useForm, FormProvider, useFormContext } from 'react-hook-form'
import { ChevronLeftIcon, ChevronRightIcon, InformationCircleIcon } from '@heroicons/react/24/outline'
import { toast } from 'react-hot-toast'

// TypeScript interfaces
interface PersonalInfo {
  age: number
  gender: 'Male' | 'Female' | 'Other'
  ethnicity: 'Caucasian' | 'African American' | 'Asian' | 'Other'
  educationLevel: 'None' | 'High School' | "Bachelor's" | 'Higher'
}

interface PhysicalHealth {
  height?: number
  weight?: number
  bmi: number
  systolicBP: number
  diastolicBP: number
  cholesterolTotal: number
  cholesterolLDL: number
  cholesterolHDL: number
  cholesterolTriglycerides: number
}

interface LifestyleFactors {
  smoking: boolean
  alcoholConsumption: number
  physicalActivity: number
  dietQuality: number
  sleepQuality: number
}

interface MedicalHistory {
  familyHistoryAlzheimers: boolean
  cardiovascularDisease: boolean
  diabetes: boolean
  depression: boolean
  headInjury: boolean
  hypertension: boolean
}

interface CognitiveAssessment {
  mmse: number
  functionalAssessment: number
  adl: number
  memoryComplaints: boolean
  behavioralProblems: boolean
  confusion: boolean
  disorientation: boolean
  personalityChanges: boolean
  difficultyCompletingTasks: boolean
  forgetfulness: boolean
}

interface FormData extends PersonalInfo, PhysicalHealth, LifestyleFactors, MedicalHistory, CognitiveAssessment {
  termsAccepted: boolean
}

const STORAGE_KEY = 'alzheimer-assessment-form'

const PatientAssessmentForm: React.FC = () => {
  const [currentStep, setCurrentStep] = useState(1)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [riskResult, setRiskResult] = useState<{score: number, risk: string} | null>(null)
  
  const methods = useForm<FormData>({
    mode: 'onChange',
    defaultValues: {
      age: 65,
      gender: 'Male',
      ethnicity: 'Caucasian',
      educationLevel: 'High School',
      bmi: 25,
      systolicBP: 120,
      diastolicBP: 80,
      cholesterolTotal: 200,
      cholesterolLDL: 100,
      cholesterolHDL: 50,
      cholesterolTriglycerides: 150,
      smoking: false,
      alcoholConsumption: 2,
      physicalActivity: 5,
      dietQuality: 5,
      sleepQuality: 7,
      familyHistoryAlzheimers: false,
      cardiovascularDisease: false,
      diabetes: false,
      depression: false,
      headInjury: false,
      hypertension: false,
      mmse: 28,
      functionalAssessment: 8,
      adl: 9,
      memoryComplaints: false,
      behavioralProblems: false,
      confusion: false,
      disorientation: false,
      personalityChanges: false,
      difficultyCompletingTasks: false,
      forgetfulness: false,
      termsAccepted: false
    }
  })

  const { watch, setValue, handleSubmit, formState: { errors, isValid } } = methods

  // Save to localStorage on form changes
  useEffect(() => {
    const subscription = watch((data) => {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
    })
    return () => subscription.unsubscribe()
  }, [watch])

  // Load from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) {
      const data = JSON.parse(saved)
      Object.keys(data).forEach(key => {
        setValue(key as keyof FormData, data[key])
      })
    }
  }, [setValue])

  // Calculate BMI automatically
  const height = watch('height')
  const weight = watch('weight')
  useEffect(() => {
    if (height && weight && height > 0) {
      const heightInMeters = height / 100
      const calculatedBMI = weight / (heightInMeters * heightInMeters)
      setValue('bmi', Math.round(calculatedBMI * 10) / 10)
    }
  }, [height, weight, setValue])

  const nextStep = () => {
    if (currentStep < 6) {
      setCurrentStep(currentStep + 1)
    }
  }

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1)
    }
  }

  const onSubmit = async (data: FormData) => {
    console.log('Form submitted with data:', data)
    console.log('Terms accepted:', data.termsAccepted)
    
    if (!data.termsAccepted) {
      toast.error('Please accept the terms and conditions')
      return
    }

    setIsSubmitting(true)
    try {
      // Transform data for API call
      const apiData = {
        Age: data.age,
        Gender: data.gender === 'Male' ? 1 : 0,
        BMI: data.bmi,
        SystolicBP: data.systolicBP,
        DiastolicBP: data.diastolicBP,
        CholesterolTotal: data.cholesterolTotal,
        CholesterolLDL: data.cholesterolLDL,
        CholesterolHDL: data.cholesterolHDL,
        CholesterolTriglycerides: data.cholesterolTriglycerides,
        Smoking: data.smoking ? 1 : 0,
        AlcoholConsumption: data.alcoholConsumption,
        PhysicalActivity: data.physicalActivity,
        DietQuality: data.dietQuality,
        SleepQuality: data.sleepQuality,
        FamilyHistoryAlzheimers: data.familyHistoryAlzheimers ? 1 : 0,
        CardiovascularDisease: data.cardiovascularDisease ? 1 : 0,
        Diabetes: data.diabetes ? 1 : 0,
        Depression: data.depression ? 1 : 0,
        HeadInjury: data.headInjury ? 1 : 0,
        Hypertension: data.hypertension ? 1 : 0,
        MMSE: data.mmse,
        FunctionalAssessment: data.functionalAssessment,
        ADL: data.adl,
        MemoryComplaints: data.memoryComplaints ? 1 : 0,
        BehavioralProblems: data.behavioralProblems ? 1 : 0,
        Confusion: data.confusion ? 1 : 0,
        Disorientation: data.disorientation ? 1 : 0,
        PersonalityChanges: data.personalityChanges ? 1 : 0,
        DifficultyCompletingTasks: data.difficultyCompletingTasks ? 1 : 0,
        Forgetfulness: data.forgetfulness ? 1 : 0
      }

      // Call prediction API
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(apiData),
      })

      if (!response.ok) {
        throw new Error('Prediction failed')
      }

      const result = await response.json()
      setRiskResult(result)
      
      // Clear saved form data
      localStorage.removeItem(STORAGE_KEY)
      toast.success('Assessment completed successfully!')
      
    } catch (error) {
      console.error('Submission error:', error)
      toast.error('Failed to complete assessment. Please try again.')
    } finally {
      setIsSubmitting(false)
    }
  }

  if (riskResult) {
    return <RiskResultDisplay result={riskResult} onReset={() => {
      setRiskResult(null)
      setCurrentStep(1)
      methods.reset()
    }} />
  }

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-xl border border-gray-200">
      <FormProvider {...methods}>
        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex justify-between items-center mb-2">
            <h2 className="text-3xl font-bold text-gray-900">Alzheimer's Risk Assessment</h2>
            <span className="text-sm text-gray-600 bg-gray-100 px-3 py-1 rounded-full font-medium">Step {currentStep} of 6</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3 shadow-inner">
            <div 
              className="bg-blue-600 h-3 rounded-full transition-all duration-500 ease-out progress-bar shadow-sm"
              style={{ width: `${(currentStep / 6) * 100}%` }}
            />
          </div>
        </div>

        <form onSubmit={handleSubmit(onSubmit)}>
          {/* Step Content */}
          {currentStep === 1 && <PersonalInfoStep />}
          {currentStep === 2 && <PhysicalHealthStep />}
          {currentStep === 3 && <LifestyleFactorsStep />}
          {currentStep === 4 && <MedicalHistoryStep />}
          {currentStep === 5 && <CognitiveAssessmentStep />}
          {currentStep === 6 && <ReviewStep />}

          {/* Navigation Buttons */}
          <div className="flex justify-between mt-8 pt-6 border-t-2 border-gray-200">
            <button
              type="button"
              onClick={prevStep}
              disabled={currentStep === 1}
              className="flex items-center px-6 py-3 text-sm font-medium text-gray-700 bg-white border-2 border-gray-300 rounded-lg hover:bg-gray-50 hover:border-gray-400 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
            >
              <ChevronLeftIcon className="w-5 h-5 mr-2" />
              Previous
            </button>

            {currentStep < 6 ? (
              <button
                type="button"
                onClick={nextStep}
                className="flex items-center px-6 py-3 text-sm font-medium text-white bg-blue-600 border-2 border-blue-600 rounded-lg hover:bg-blue-700 hover:border-blue-700 transition-all duration-200 shadow-md hover:shadow-lg"
              >
                Next
                <ChevronRightIcon className="w-5 h-5 ml-2" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={isSubmitting}
                className="flex items-center px-8 py-3 text-sm font-medium text-white bg-green-600 border-2 border-green-600 rounded-lg hover:bg-green-700 hover:border-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-md hover:shadow-lg"
              >
                {isSubmitting ? (
                  <>
                    <div className="w-5 h-5 mr-2 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <span className="mr-2">🧬</span>
                    Submit Assessment
                  </>
                )}
              </button>
            )}
          </div>
        </form>
      </FormProvider>
    </div>
  )
}

// Step Components
const PersonalInfoStep: React.FC = () => {
  const { register, formState: { errors } } = useFormContext<FormData>()

  return (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold text-gray-900">Personal Information</h3>
      
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
        <div>
          <label className="block text-sm font-medium text-gray-900 mb-2">
            Age
          </label>
          <input
            type="number"
            min="18"
            max="120"
            {...register('age', { 
              required: 'Age is required',
              min: { value: 18, message: 'Age must be at least 18' },
              max: { value: 120, message: 'Age must be less than 120' }
            })}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900 bg-white"
          />
          {errors.age && <p className="mt-1 text-sm text-red-600">{errors.age.message}</p>}
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-900 mb-2">
            Gender
          </label>
          <div className="space-y-2">
            {['Male', 'Female', 'Other'].map(option => (
              <label key={option} className="flex items-center">
                <input
                  type="radio"
                  value={option}
                  {...register('gender', { required: 'Gender is required' })}
                  className="mr-2 text-blue-600"
                />
                <span className="text-gray-900">{option}</span>
              </label>
            ))}
          </div>
          {errors.gender && <p className="mt-1 text-sm text-red-600">{errors.gender.message}</p>}
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-900 mb-2">
            Ethnicity
          </label>
          <select
            {...register('ethnicity', { required: 'Ethnicity is required' })}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900 bg-white"
          >
            <option value="Caucasian">Caucasian</option>
            <option value="African American">African American</option>
            <option value="Asian">Asian</option>
            <option value="Other">Other</option>
          </select>
          {errors.ethnicity && <p className="mt-1 text-sm text-red-600">{errors.ethnicity.message}</p>}
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-900 mb-2">
            Education Level
          </label>
          <select
            {...register('educationLevel', { required: 'Education level is required' })}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900 bg-white"
          >
            <option value="None">None</option>
            <option value="High School">High School</option>
            <option value="Bachelor's">Bachelor's</option>
            <option value="Higher">Higher</option>
          </select>
          {errors.educationLevel && <p className="mt-1 text-sm text-red-600">{errors.educationLevel.message}</p>}
        </div>
      </div>
    </div>
  )
}

const PhysicalHealthStep: React.FC = () => {
  const { register, watch, formState: { errors } } = useFormContext<FormData>()
  const bmi = watch('bmi')

  const getBMICategory = (bmi: number) => {
    if (bmi < 18.5) return { category: 'Underweight', color: 'text-blue-600' }
    if (bmi < 25) return { category: 'Normal', color: 'text-green-600' }
    if (bmi < 30) return { category: 'Overweight', color: 'text-yellow-600' }
    return { category: 'Obese', color: 'text-red-600' }
  }

  return (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold text-gray-900">Physical Health</h3>
      
      {/* BMI Calculator */}
      <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
        <h4 className="font-medium text-gray-900 mb-3">BMI Calculator</h4>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <div>
            <label className="block text-sm font-medium text-gray-900 mb-1">
              Height (cm)
            </label>
            <input
              type="number"
              min="100"
              max="250"
              {...register('height')}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900 bg-white"
              placeholder="170"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-900 mb-1">
              Weight (kg)
            </label>
            <input
              type="number"
              min="30"
              max="300"
              {...register('weight')}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900 bg-white"
              placeholder="70"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-900 mb-1">
              BMI
            </label>
            <div className="flex items-center space-x-2">
              <input
                type="number"
                step="0.1"
                min="10"
                max="50"
                {...register('bmi', { 
                  required: 'BMI is required',
                  min: { value: 10, message: 'BMI too low' },
                  max: { value: 50, message: 'BMI too high' }
                })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900 bg-white"
              />
              {bmi && (
                <span className={`text-sm font-medium ${getBMICategory(bmi).color}`}>
                  {getBMICategory(bmi).category}
                </span>
              )}
            </div>
            {errors.bmi && <p className="mt-1 text-sm text-red-600">{errors.bmi.message}</p>}
          </div>
        </div>
      </div>

      {/* Blood Pressure */}
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
        <div>
          <label className="block text-sm font-medium text-gray-900 mb-2">
            Systolic Blood Pressure (mmHg)
            <Tooltip content="The top number in blood pressure reading" />
          </label>
          <input
            type="number"
            min="80"
            max="200"
            {...register('systolicBP', { 
              required: 'Systolic BP is required',
              min: { value: 80, message: 'Value too low' },
              max: { value: 200, message: 'Value too high' }
            })}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900 bg-white"
          />
          {errors.systolicBP && <p className="mt-1 text-sm text-red-600">{errors.systolicBP.message}</p>}
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-900 mb-2">
            Diastolic Blood Pressure (mmHg)
            <Tooltip content="The bottom number in blood pressure reading" />
          </label>
          <input
            type="number"
            min="40"
            max="130"
            {...register('diastolicBP', { 
              required: 'Diastolic BP is required',
              min: { value: 40, message: 'Value too low' },
              max: { value: 130, message: 'Value too high' }
            })}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900 bg-white"
          />
          {errors.diastolicBP && <p className="mt-1 text-sm text-red-600">{errors.diastolicBP.message}</p>}
        </div>
      </div>

      {/* Cholesterol */}
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
        <div>
          <label className="block text-sm font-medium text-gray-900 mb-2">
            Total Cholesterol (mg/dL)
          </label>
          <input
            type="number"
            min="100"
            max="400"
            {...register('cholesterolTotal', { 
              required: 'Total cholesterol is required',
              min: { value: 100, message: 'Value too low' },
              max: { value: 400, message: 'Value too high' }
            })}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900 bg-white"
          />
          {errors.cholesterolTotal && <p className="mt-1 text-sm text-red-600">{errors.cholesterolTotal.message}</p>}
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-900 mb-2">
            LDL Cholesterol (mg/dL)
            <Tooltip content="'Bad' cholesterol" />
          </label>
          <input
            type="number"
            min="50"
            max="250"
            {...register('cholesterolLDL', { 
              required: 'LDL cholesterol is required',
              min: { value: 50, message: 'Value too low' },
              max: { value: 250, message: 'Value too high' }
            })}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900 bg-white"
          />
          {errors.cholesterolLDL && <p className="mt-1 text-sm text-red-600">{errors.cholesterolLDL.message}</p>}
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-900 mb-2">
            HDL Cholesterol (mg/dL)
            <Tooltip content="'Good' cholesterol" />
          </label>
          <input
            type="number"
            min="20"
            max="100"
            {...register('cholesterolHDL', { 
              required: 'HDL cholesterol is required',
              min: { value: 20, message: 'Value too low' },
              max: { value: 100, message: 'Value too high' }
            })}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900 bg-white"
          />
          {errors.cholesterolHDL && <p className="mt-1 text-sm text-red-600">{errors.cholesterolHDL.message}</p>}
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-900 mb-2">
            Triglycerides (mg/dL)
          </label>
          <input
            type="number"
            min="50"
            max="500"
            {...register('cholesterolTriglycerides', { 
              required: 'Triglycerides is required',
              min: { value: 50, message: 'Value too low' },
              max: { value: 500, message: 'Value too high' }
            })}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900 bg-white"
          />
          {errors.cholesterolTriglycerides && <p className="mt-1 text-sm text-red-600">{errors.cholesterolTriglycerides.message}</p>}
        </div>
      </div>
    </div>
  )
}

const LifestyleFactorsStep: React.FC = () => {
  const { register, watch, setValue } = useFormContext<FormData>()
  
  const alcoholConsumption = watch('alcoholConsumption')
  const physicalActivity = watch('physicalActivity')
  const dietQuality = watch('dietQuality')
  const sleepQuality = watch('sleepQuality')

  return (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold text-gray-900">Lifestyle Factors</h3>
      
      {/* Smoking */}
      <div className="bg-gray-50 p-4 rounded-lg border">
        <label className="flex items-center space-x-3 cursor-pointer">
          <input
            type="checkbox"
            {...register('smoking')}
            className="w-5 h-5 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
          />
          <span className="text-sm font-medium text-gray-900">
            Current smoker
          </span>
          <span className="text-2xl">🚬</span>
        </label>
      </div>

      {/* Sliders */}
      <div className="space-y-6">
        <div className="bg-white p-4 rounded-lg border">
          <label className="block text-sm font-medium text-gray-900 mb-3">
            Alcohol Consumption (drinks per week): <span className="text-blue-600 font-bold">{alcoholConsumption}</span>
          </label>
          <input
            type="range"
            min="0"
            max="10"
            {...register('alcoholConsumption')}
            className="w-full h-3 bg-gray-200 rounded-lg appearance-none cursor-pointer slider-thumb"
          />
          <div className="flex justify-between text-xs text-gray-600 mt-2 font-medium">
            <span>None</span>
            <span>Moderate</span>
            <span>Heavy</span>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg border">
          <label className="block text-sm font-medium text-gray-900 mb-3">
            Physical Activity Level: <span className="text-blue-600 font-bold">{physicalActivity}/10</span>
          </label>
          <input
            type="range"
            min="0"
            max="10"
            {...register('physicalActivity')}
            className="w-full h-3 bg-gray-200 rounded-lg appearance-none cursor-pointer slider-thumb"
          />
          <div className="flex justify-between text-xs text-gray-600 mt-2 font-medium">
            <span>Sedentary</span>
            <span>Moderate</span>
            <span>Very Active</span>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg border">
          <label className="block text-sm font-medium text-gray-900 mb-3">
            Diet Quality: <span className="text-blue-600 font-bold">{dietQuality}/10</span>
          </label>
          <input
            type="range"
            min="0"
            max="10"
            {...register('dietQuality')}
            className="w-full h-3 bg-gray-200 rounded-lg appearance-none cursor-pointer slider-thumb"
          />
          <div className="flex justify-between text-xs text-gray-600 mt-2 font-medium">
            <span>Poor</span>
            <span>Good</span>
            <span>Excellent</span>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg border">
          <label className="block text-sm font-medium text-gray-900 mb-3">
            Sleep Quality: <span className="text-blue-600 font-bold">{sleepQuality}/10</span>
          </label>
          <input
            type="range"
            min="0"
            max="10"
            {...register('sleepQuality')}
            className="w-full h-3 bg-gray-200 rounded-lg appearance-none cursor-pointer slider-thumb"
          />
          <div className="flex justify-between text-xs text-gray-600 mt-2 font-medium">
            <span>Poor</span>
            <span>Good</span>
            <span>Excellent</span>
          </div>
        </div>
      </div>
    </div>
  )
}

const MedicalHistoryStep: React.FC = () => {
  const { register } = useFormContext<FormData>()

  const conditions = [
    { name: 'familyHistoryAlzheimers', label: 'Family History of Alzheimer\'s', icon: '🧠' },
    { name: 'cardiovascularDisease', label: 'Cardiovascular Disease', icon: '❤️' },
    { name: 'diabetes', label: 'Diabetes', icon: '🍯' },
    { name: 'depression', label: 'Depression', icon: '😔' },
    { name: 'headInjury', label: 'Head Injury', icon: '🤕' },
    { name: 'hypertension', label: 'Hypertension', icon: '🩺' },
  ]

  return (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold text-gray-900">Medical History</h3>
      
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {conditions.map(condition => (
          <label key={condition.name} className="flex items-center space-x-3 p-4 border-2 border-gray-200 rounded-lg hover:bg-blue-50 hover:border-blue-300 cursor-pointer transition-all">
            <input
              type="checkbox"
              {...register(condition.name as keyof FormData)}
              className="w-5 h-5 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
            />
            <span className="text-2xl">{condition.icon}</span>
            <span className="text-sm font-medium text-gray-900">
              {condition.label}
            </span>
            {condition.name === 'familyHistoryAlzheimers' && (
              <Tooltip content="Having a family member with Alzheimer's increases risk" />
            )}
          </label>
        ))}
      </div>
    </div>
  )
}

const CognitiveAssessmentStep: React.FC = () => {
  const { register, watch, formState: { errors } } = useFormContext<FormData>()
  
  const mmse = watch('mmse')
  const functionalAssessment = watch('functionalAssessment')
  const adl = watch('adl')

  const cognitiveSymptoms = [
    { name: 'memoryComplaints', label: 'Memory Complaints', icon: '🧠' },
    { name: 'behavioralProblems', label: 'Behavioral Problems', icon: '😤' },
    { name: 'confusion', label: 'Confusion', icon: '😵' },
    { name: 'disorientation', label: 'Disorientation', icon: '🌀' },
    { name: 'personalityChanges', label: 'Personality Changes', icon: '🎭' },
    { name: 'difficultyCompletingTasks', label: 'Difficulty Completing Tasks', icon: '📝' },
    { name: 'forgetfulness', label: 'Forgetfulness', icon: '🤔' },
  ]

  return (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold text-gray-900">Cognitive Assessment</h3>
      
      {/* MMSE Score */}
      <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
        <label className="block text-sm font-medium text-gray-900 mb-2">
          MMSE Score (0-30)
          <Tooltip content="Mini-Mental State Examination: 24-30 = Normal, 18-23 = Mild cognitive impairment, <18 = Severe" />
        </label>
        <input
          type="number"
          min="0"
          max="30"
          {...register('mmse', { 
            required: 'MMSE score is required',
            min: { value: 0, message: 'Score must be at least 0' },
            max: { value: 30, message: 'Score must be at most 30' }
          })}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900 bg-white"
        />
        {mmse && (
          <p className="mt-2 text-sm font-medium">
            {mmse >= 24 ? '✅ Normal cognitive function' : mmse >= 18 ? '⚠️ Mild cognitive impairment' : '🚨 Severe cognitive impairment'}
          </p>
        )}
        {errors.mmse && <p className="mt-1 text-sm text-red-600">{errors.mmse.message}</p>}
      </div>

      {/* Assessment Sliders */}
      <div className="space-y-4">
        <div className="bg-white p-4 rounded-lg border">
          <label className="block text-sm font-medium text-gray-900 mb-3">
            Functional Assessment: <span className="text-blue-600 font-bold">{functionalAssessment}/10</span>
          </label>
          <input
            type="range"
            min="0"
            max="10"
            {...register('functionalAssessment')}
            className="w-full h-3 bg-gray-200 rounded-lg appearance-none cursor-pointer slider-thumb"
          />
          <div className="flex justify-between text-xs text-gray-600 mt-2 font-medium">
            <span>Poor</span>
            <span>Good</span>
            <span>Excellent</span>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg border">
          <label className="block text-sm font-medium text-gray-900 mb-3">
            Activities of Daily Living (ADL): <span className="text-blue-600 font-bold">{adl}/10</span>
          </label>
          <input
            type="range"
            min="0"
            max="10"
            {...register('adl')}
            className="w-full h-3 bg-gray-200 rounded-lg appearance-none cursor-pointer slider-thumb"
          />
          <div className="flex justify-between text-xs text-gray-600 mt-2 font-medium">
            <span>Dependent</span>
            <span>Some Help</span>
            <span>Independent</span>
          </div>
        </div>
      </div>

      {/* Cognitive Symptoms */}
      <div>
        <h4 className="text-lg font-medium text-gray-900 mb-4">Cognitive Symptoms</h4>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {cognitiveSymptoms.map(symptom => (
            <label key={symptom.name} className="flex items-center space-x-3 p-4 border-2 border-gray-200 rounded-lg hover:bg-red-50 hover:border-red-300 cursor-pointer transition-all">
              <input
                type="checkbox"
                {...register(symptom.name as keyof FormData)}
                className="w-5 h-5 text-red-600 border-gray-300 rounded focus:ring-red-500"
              />
              <span className="text-xl">{symptom.icon}</span>
              <span className="text-sm font-medium text-gray-900">
                {symptom.label}
              </span>
            </label>
          ))}
        </div>
      </div>
    </div>
  )
}

const ReviewStep: React.FC = () => {
  const { register, watch, formState: { errors } } = useFormContext<FormData>()
  const formData = watch()

  const sections = [
    {
      title: 'Personal Information',
      icon: '👤',
      data: {
        'Age': formData.age,
        'Gender': formData.gender,
        'Ethnicity': formData.ethnicity,
        'Education': formData.educationLevel
      }
    },
    {
      title: 'Physical Health',
      icon: '🏥',
      data: {
        'BMI': formData.bmi,
        'Blood Pressure': `${formData.systolicBP}/${formData.diastolicBP}`,
        'Total Cholesterol': formData.cholesterolTotal,
        'LDL': formData.cholesterolLDL,
        'HDL': formData.cholesterolHDL,
        'Triglycerides': formData.cholesterolTriglycerides
      }
    },
    {
      title: 'Lifestyle',
      icon: '🏃‍♂️',
      data: {
        'Smoking': formData.smoking ? 'Yes' : 'No',
        'Alcohol (drinks/week)': formData.alcoholConsumption,
        'Physical Activity': `${formData.physicalActivity}/10`,
        'Diet Quality': `${formData.dietQuality}/10`,
        'Sleep Quality': `${formData.sleepQuality}/10`
      }
    },
    {
      title: 'Medical History',
      icon: '📋',
      data: {
        'Family History': formData.familyHistoryAlzheimers ? 'Yes' : 'No',
        'Cardiovascular Disease': formData.cardiovascularDisease ? 'Yes' : 'No',
        'Diabetes': formData.diabetes ? 'Yes' : 'No',
        'Depression': formData.depression ? 'Yes' : 'No',
        'Head Injury': formData.headInjury ? 'Yes' : 'No',
        'Hypertension': formData.hypertension ? 'Yes' : 'No'
      }
    },
    {
      title: 'Cognitive Assessment',
      icon: '🧠',
      data: {
        'MMSE Score': `${formData.mmse}/30`,
        'Functional Assessment': `${formData.functionalAssessment}/10`,
        'ADL': `${formData.adl}/10`,
        'Memory Complaints': formData.memoryComplaints ? 'Yes' : 'No',
        'Confusion': formData.confusion ? 'Yes' : 'No',
        'Forgetfulness': formData.forgetfulness ? 'Yes' : 'No'
      }
    }
  ]

  return (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold text-gray-900">Review Your Information</h3>
      
      {sections.map((section, index) => (
        <div key={index} className="bg-linear-to-r from-blue-50 to-indigo-50 p-6 rounded-lg border border-blue-200">
          <div className="flex items-center mb-4">
            <span className="text-2xl mr-3">{section.icon}</span>
            <h4 className="font-semibold text-gray-900 text-lg">{section.title}</h4>
          </div>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {Object.entries(section.data).map(([key, value]) => (
              <div key={key} className="flex justify-between bg-white p-3 rounded border">
                <span className="text-sm font-medium text-gray-700">{key}:</span>
                <span className="text-sm font-bold text-gray-900">{value}</span>
              </div>
            ))}
          </div>
        </div>
      ))}

      {/* Terms and Conditions */}
      <div className="border-t-2 border-gray-200 pt-6">
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
          <h4 className="font-semibold text-yellow-800 mb-2">⚠️ Important Disclaimer</h4>
          <p className="text-sm text-yellow-700">
            This assessment is for educational and informational purposes only. Results should not replace professional medical advice, diagnosis, or treatment.
          </p>
        </div>
        
        <label className="flex items-start space-x-3 cursor-pointer">
          <input
            type="checkbox"
            {...register('termsAccepted', { required: 'You must accept the terms and conditions' })}
            className="w-5 h-5 text-blue-600 border-gray-300 rounded focus:ring-blue-500 mt-1"
          />
          <div className="text-sm">
            <span className="text-gray-900 font-medium">
              I agree to the terms and conditions and understand that this assessment is for informational purposes only and should not replace professional medical advice.
            </span>
            {/* Debug info */}
            <div className="mt-2 text-xs text-gray-500">
              Terms Accepted: {formData.termsAccepted ? 'Yes' : 'No'}
            </div>
          </div>
        </label>
        {errors.termsAccepted && <p className="mt-2 text-sm text-red-600 font-medium">{errors.termsAccepted.message}</p>}
      </div>
    </div>
  )
}

// Risk Result Display Component
const RiskResultDisplay: React.FC<{ result: {score: number, risk: string}, onReset: () => void }> = ({ result, onReset }) => {
  const getRiskColor = (risk: string) => {
    switch (risk.toLowerCase()) {
      case 'low': return 'text-green-700 bg-green-100 border-green-300'
      case 'moderate': return 'text-yellow-700 bg-yellow-100 border-yellow-300'
      case 'high': return 'text-red-700 bg-red-100 border-red-300'
      default: return 'text-gray-700 bg-gray-100 border-gray-300'
    }
  }

  const getRiskIcon = (risk: string) => {
    switch (risk.toLowerCase()) {
      case 'low': return '✅'
      case 'moderate': return '⚠️'
      case 'high': return '🚨'
      default: return '❓'
    }
  }

  return (
    <div className="max-w-3xl mx-auto p-8 bg-white rounded-xl shadow-2xl border border-gray-200">
      <div className="text-center mb-8">
        <div className="mb-6">
          <h2 className="text-4xl font-bold text-gray-900 mb-2">Assessment Complete</h2>
          <p className="text-lg text-gray-600">Your Alzheimer's Risk Analysis</p>
        </div>
        
        <div className="mb-8 p-6 bg-gray-50 rounded-xl">
          <div className="text-7xl font-bold text-blue-600 mb-3">
            {Math.round(result.score * 100)}%
          </div>
          <div className="text-xl text-gray-700 font-medium">Risk Probability</div>
        </div>

        <div className={`inline-flex items-center px-8 py-4 rounded-full text-2xl font-bold border-2 ${getRiskColor(result.risk)}`}>
          <span className="mr-3 text-3xl">{getRiskIcon(result.risk)}</span>
          {result.risk.toUpperCase()} RISK
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 p-6 rounded-xl mb-8">
        <h3 className="font-bold text-blue-900 mb-3 text-lg flex items-center">
          <span className="mr-2">⚕️</span>
          Important Medical Notice
        </h3>
        <p className="text-blue-800">
          This assessment uses machine learning algorithms trained on clinical data but is for informational purposes only. 
          Please consult with a qualified healthcare provider for proper diagnosis, treatment, and medical advice.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <button
          onClick={onReset}
          className="w-full px-6 py-4 text-white bg-blue-600 rounded-xl hover:bg-blue-700 font-semibold text-lg transition-all duration-200 shadow-md hover:shadow-lg flex items-center justify-center"
        >
          <span className="mr-2">🔄</span>
          Take Another Assessment
        </button>
        
        <button
          onClick={() => window.print()}
          className="w-full px-6 py-4 text-blue-600 bg-white border-2 border-blue-600 rounded-xl hover:bg-blue-50 font-semibold text-lg transition-all duration-200 shadow-md hover:shadow-lg flex items-center justify-center"
        >
          <span className="mr-2">🖨️</span>
          Print Results
        </button>
      </div>
    </div>
  )
}

// Tooltip Component
const Tooltip: React.FC<{ content: string }> = ({ content }) => {
  const [isVisible, setIsVisible] = useState(false)

  return (
    <div className="relative inline-block ml-1">
      <InformationCircleIcon
        className="w-5 h-5 text-blue-500 cursor-help hover:text-blue-600 transition-colors"
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
      />
      {isVisible && (
        <div className="absolute z-20 w-64 p-3 text-xs text-white bg-gray-900 rounded-lg shadow-xl -top-2 left-6 border border-gray-700">
          {content}
          <div className="absolute w-2 h-2 bg-gray-900 transform rotate-45 -left-1 top-4 border-l border-b border-gray-700"></div>
        </div>
      )}
    </div>
  )
}

export default PatientAssessmentForm