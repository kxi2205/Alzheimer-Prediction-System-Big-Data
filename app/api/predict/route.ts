import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

export async function POST(request: NextRequest) {
  try {
    const data = await request.json()
    
    // Validate required fields
    const requiredFields = [
      'Age', 'Gender', 'BMI', 'SystolicBP', 'DiastolicBP',
      'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
      'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
      'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression',
      'HeadInjury', 'Hypertension', 'MMSE', 'FunctionalAssessment', 'ADL',
      'MemoryComplaints', 'BehavioralProblems', 'Confusion', 'Disorientation',
      'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness'
    ]

    for (const field of requiredFields) {
      if (data[field] === undefined || data[field] === null) {
        return NextResponse.json(
          { error: `Missing required field: ${field}` },
          { status: 400 }
        )
      }
    }

    // Call Python prediction script
    const pythonScriptPath = path.join(process.cwd(), 'api', 'python', 'predict.py')
    const pythonExecutable = path.join(process.cwd(), '.venv', 'Scripts', 'python.exe')
    
    const result = await new Promise((resolve, reject) => {
      const pythonProcess = spawn(pythonExecutable, [pythonScriptPath], {
        stdio: ['pipe', 'pipe', 'pipe']
      })

      let stdout = ''
      let stderr = ''

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString()
      })

      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString()
      })

      pythonProcess.on('close', (code) => {
        console.log('Python script stdout:', stdout)
        console.log('Python script stderr:', stderr)
        console.log('Python script exit code:', code)
        
        if (code !== 0) {
          reject(new Error(`Python script failed with code ${code}: ${stderr}`))
        } else {
          try {
            const result = JSON.parse(stdout.trim())
            resolve(result)
          } catch (e) {
            console.error('Failed to parse Python output:', stdout)
            reject(new Error('Failed to parse Python script output'))
          }
        }
      })

      // Send input data to Python script
      pythonProcess.stdin.write(JSON.stringify(data))
      pythonProcess.stdin.end()
    })

    return NextResponse.json(result)

  } catch (error) {
    console.error('Prediction API error:', error)
    return NextResponse.json(
      { error: 'Internal server error during prediction' },
      { status: 500 }
    )
  }
}