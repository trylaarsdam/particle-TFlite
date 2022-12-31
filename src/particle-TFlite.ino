SYSTEM_MODE(MANUAL);
/*
 * Project particle-TFlite
 * Description:
 * Author:
 * Date:
 */

#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "linreg_model_data.hpp"

const tflite::Model *model = tflite::GetModel(g_linear_regresion_model_data);
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

float randFloat(float min, float max)
{
  return ((max - min) * ((float)rand() / RAND_MAX)) + min;
}

static tflite::ops::micro::AllOpsResolver resolver;
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
																		kTensorArenaSize, error_reporter);
TfLiteStatus allocate_status = interpreter.AllocateTensors();

// setup() runs once, when the device is first turned on.
void setup()
{
	// Put initialization like pinMode and begin functions here.
	Serial.begin();
	if (model->version() != TFLITE_SCHEMA_VERSION)
	{
		error_reporter->Report(
			"Model provided is schema version %d not equal "
			"to supported version %d.",
			model->version(), TFLITE_SCHEMA_VERSION);
		return;
	}

	

	input = interpreter.input(0);
	output = interpreter.output(0);
}

// loop() runs over and over again, as quickly as it can execute.
void loop()
{
	float x_val = randFloat(0, 1);
	input->data.f[0] = x_val;

	TfLiteStatus invoke_status = interpreter.Invoke();
	if (invoke_status != kTfLiteOk)
	{
		error_reporter->Report("Invoke failed on x_val: %f\n",
														static_cast<double>(x_val));
		return;
	}

	float y_val = output->data.f[0];

	Serial.printlnf("%.2f, %.2f", x_val, y_val);
	delay(500);
}