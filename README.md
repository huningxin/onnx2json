# onnx2webnn
Exports the ONNX file to a WebNN JavaScript file and a bin file containing the weights.

This project is derived from [onnx2json](https://github.com/PINTO0309/onnx2json).

## Setup

```bash
$ pip install -U onnx protobuf numpy
```
## CLI Usage
```
usage: onnx2webnn.py [-h] -if INPUT_ONNX_FILE_PATH -oj OUTPUT_JS_PATH [-nhwc] [-json] [-i JSON_INDENT]

options:
  -h, --help            show this help message and exit
  -if INPUT_ONNX_FILE_PATH, --input_onnx_file_path INPUT_ONNX_FILE_PATH
                        Input ONNX model path. (*.onnx)
  -oj OUTPUT_JS_PATH, --output_js_path OUTPUT_JS_PATH
                        Output WebNN JavaScript file path (*.js)
  -nhwc, --nhwc         Generate WebNN operators taking nhwc input layout, including conv2d, convTranspose2d, resample2d and pool2d
  -json, --dump_json    Dump the JSON representation of ONNX model
  -i JSON_INDENT, --json_indent JSON_INDENT
                        Number of indentations in JSON. (default=2)
```

## Generate WebNN JavaScript model
*Note*: Before using this tool, please ensure override the free dimensions of input ONNX model by using [onnxruntime_perf_test](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/perftest/README.md) tool, for example
```bash
$ onnxruntime_perf_test -I -r 1 -u mobilenetv2-12-static.onnx -f batch_size:1 -o 1 mobilenetv2-12.onnx
```

Make a folder "mobilenet" that will contain the generated files:
```bash
$ mkdir mobilenet
```

Then run the following command to create WebNN JavaScript model for the static ONNX model:
```bash
$ python onnx2webnn.py -if ../sample_models/mobilenetv2-12-static.onnx -oj mobilenet/mobilenet.js
```
It will generate "mobilenet.bin" and "mobilenet.js" besides "mobilenet.json" in "mobilenet" folder.

An "index.html" is also generated for testing the WebNN model.

Start a node.js http-server in the folder containing generated model files and launch a web browser with URL http://localhost:8080/.
```bash
$ http-server
```

## Generate NHWC WebNN model
The default input layout of ONNX model is NCHW, however some WebNN backends prefer to NHWC input layout, such as TFLite for CPU and GPU. For those backends, using WebNN NHWC model may have better performance.

To generate the NHWC WebNN model from an ONNX model, add the "--nhwc" switch, such as
```bash
$ python onnx2webnn.py -if ../sample_models/mobilenetv2-12-static.onnx -oj mobilenet_nhwc/mobilenet_nhwc.js -nhwc
```

WebNN API exposes the backend preferred layout via [`MLContext.opSupportLimit().preferredInputLayout`](https://www.w3.org/TR/webnn/#dom-mlopsupportlimits-preferredinputlayout), a web app can load the corresponding WebNN model based on the preferred layout. For example, in JavaScript code
```javascript
const deviceType = 'gpu'; // or 'cpu', 'npu'
const context = await navigator.ml.createContext({deviceType});
const layout = context.opSupportLimits().preferredInputLayout;
let webnnModel;
if (layout == 'nhwc') {
    webnnModel = new MobilenetNhwc();
} else {
    webnnModel = new Mobilenet();
}
// Load the weights in preferred layout and build the graph.
await webnnModel.build({deviceType});
// Do inference with webnnModel.run()
```

## Dump JSON
You can also dump the JSON file for debugging purpose.
```bash
$ python onnx2webnn.py -if ../sample_models/mobilenetv2-12-static.onnx -oj mobilenet/mobilenet.js -json
```
