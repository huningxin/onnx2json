#! /usr/bin/env python

import sys
import onnx
import json
import os
from google.protobuf.json_format import MessageToJson
from typing import Optional
from argparse import ArgumentParser

class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'


def convert(
    input_onnx_file_path: Optional[str] = '',
    onnx_graph: Optional[onnx.ModelProto] = None,
    output_json_path: Optional[str] = '',
    json_indent: Optional[int] = 2,
    external_weights: Optional[bool] = False,
    webnn_js: Optional[bool] = False,
):
    """
    Parameters
    ----------
    input_onnx_file_path: Optional[str]
        Input onnx file path.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        Default: ''

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    output_json_path: Optional[str]
        Output JSON file path (*.json) If not specified, no JSON file is output.\n\
        Default: ''

    json_indent: Optional[int]
        Number of indentations in JSON.\n\
        Default: 2

    external_weights: Optional[bool]
        Save weights to an external file.\n\
        Default: False

    webnn_js: Optional[bool]
        Generate WebNN JavaScript code, must be used together with external_weights.\n\
        Default: False

    Returns
    -------
    onnx_json: dict
        Converted JSON dict.
    """

    # Unspecified check for input_onnx_file_path and onnx_graph
    if not input_onnx_file_path and not onnx_graph:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'One of input_onnx_file_path or onnx_graph must be specified.'
        )
        sys.exit(1)

    if not external_weights and webnn_js:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'Saving to external weights file must be used together with generating WebNN JavaScript.'
        )
        sys.exit(1)

    if not onnx_graph:
        onnx_graph = onnx.load(input_onnx_file_path)

    if not external_weights:
        s = MessageToJson(onnx_graph)
        onnx_json = json.loads(s)

        if output_json_path:
            with open(output_json_path, 'w') as f:
                json.dump(onnx_json, f, indent=json_indent)
    else:
        external_data_file_path = os.path.splitext(output_json_path)[0] + ".bin"
        # Remove existing external data file before saving
        if os.path.exists(external_data_file_path):
            os.remove(external_data_file_path)
        temp_external_data_onnx_file_path = "temp_external_data_model.onnx"
        onnx.save_model(onnx_graph, temp_external_data_onnx_file_path, save_as_external_data=True, all_tensors_to_one_file=True, location=external_data_file_path, size_threshold=0, convert_attribute=False)
        external_data_model = onnx.load(temp_external_data_onnx_file_path, load_external_data=False)
        os.remove(temp_external_data_onnx_file_path)
        s = MessageToJson(external_data_model)
        onnx_json = json.loads(s)

        if output_json_path:
            with open(output_json_path, 'w') as f:
                json.dump(onnx_json, f, indent=json_indent)

        if not webnn_js:
            return onnx_json

        # Generate WebNN JavaScript code
        js_lines = []

        def to_js_var_name(name: str) -> str:
            """
            Convert a string to a legal JavaScript variable name.
            If the name is not a valid identifier or starts with a digit,
            prefix with 'var_' and replace invalid characters with '_'.
            """
            js_var_name = name
            if not js_var_name.isidentifier() or js_var_name[0].isdigit():
                js_var_name = f"var_{js_var_name}"
            js_var_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in js_var_name)
            return js_var_name

        # Start Model class
        class_name = os.path.splitext(os.path.basename(output_json_path))[0]
        # Convert to valid JS class name (capitalize, remove invalid chars)
        class_name = ''.join([w.capitalize() for w in class_name.replace('-', '_').split('_') if w])
        js_lines.append(f"export class {class_name} {{")
        js_lines.append("  constructor() {")
        js_lines.append("    this.graph_ = null;")
        js_lines.append("    this.context_ = null;")
        js_lines.append("    this.inputTensors_ = {};")
        js_lines.append("    this.outputTensors_ = {};")
        js_lines.append("  }")
        js_lines.append("")
        js_lines.append("  async build(contextOptions) {")

        # JavaScript code to load weights
        js_lines.append(f"""    // Load weights ArrayBuffer from {os.path.basename(external_data_file_path)}
    async function loadWeightsArrayBuffer() {{
        const response = await fetch('{os.path.basename(external_data_file_path)}');
        if (!response.ok) {{
            throw new Error('Failed to fetch weights: ' + response.statusText);
        }}
        return await response.arrayBuffer();
    }}

    const weights_array_buffer = await loadWeightsArrayBuffer();

    this.context_ = await navigator.ml.createContext(contextOptions);
    const builder = new MLGraphBuilder(this.context_);
""")

        webnn_type_map = {
            1: 'float32',   # FLOAT
            2: 'uint8',     # UINT8
            3: 'int8',      # INT8
            4: 'uint16',    # UINT16
            5: 'int16',     # INT16
            6: 'int32',     # INT32
            7: 'int64',     # INT64 (not directly supported in JS)
            9: 'uint8',     # BOOL (special handling)
            10: 'float16',  # FLOAT16 (not directly supported in JS)
            11: 'float64',  # DOUBLE
            12: 'uint32',   # UINT32
            13: 'uint64',   # UINT64 (not directly supported in JS)
        }

        # Map ONNX data type to WebNN typed array
        typed_array_map = {
            1: 'Float32Array',   # FLOAT
            2: 'Uint8Array',     # UINT8
            3: 'Int8Array',      # INT8
            4: 'Uint16Array',    # UINT16
            5: 'Int16Array',     # INT16
            6: 'Int32Array',     # INT32
            7: 'BigInt64Array',     # INT64 (not directly supported in JS)
            9: 'Bool',           # BOOL (special handling)
            10: 'Float16Array',  # FLOAT16 (not directly supported in JS)
            11: 'Float64Array',  # DOUBLE
            12: 'Uint32Array',   # UINT32
            13: 'BigUint64Array',   # UINT64 (not directly supported in JS)
        }

        # Generate all the graph input operands and tensors
        js_lines.append("    // Create graph input operands and tensors.")
        graph_inputs = onnx_json['graph'].get('input', [])
        for input_info in graph_inputs:
            name = input_info['name']
            dims = input_info.get('type', {}).get('tensorType', {}).get('shape', {}).get('dim', [])
            dims_str = ', '.join(str(d.get('dimValue', 1)) for d in dims)
            elem_type = input_info.get('type', {}).get('tensorType', {}).get('elemType', 1)
            webnn_dtype = webnn_type_map.get(elem_type, 'float32')
            js_var_name = to_js_var_name(name)
            js_code = f"""const {js_var_name} = builder.input(
        '{name}',
        {{dataType: '{webnn_dtype}', shape: [{dims_str}]}}
    );
    this.inputTensors_['{name}'] = await this.context_.createTensor(
        {{dataType: '{webnn_dtype}', shape: [{dims_str}], writable: true}}
    );"""
            js_lines.append("    " + js_code)

        # Generate all the constant operands
        js_lines.append("    // Create graph constant operands.")
        initializers = onnx_json['graph'].get('initializer', [])
        for initializer in initializers:
            name = initializer['name']
            dims = initializer.get('dims', [])
            data_type = initializer.get('dataType', None)
            offset = None
            length = None
            isExternalData = False
            if "externalData" in initializer:
                isExternalData = True
                for entry in initializer["externalData"]:
                    if entry["key"] == "offset":
                        offset = int(entry["value"])
                    elif entry["key"] == "length":
                        length = int(entry["value"])
            webnn_dtype = webnn_type_map.get(data_type, 'float32')
            typed_array = typed_array_map.get(data_type, 'Float32Array')
            dims_str = ', '.join(str(d) for d in dims)
            if isExternalData:
                js_var_name = to_js_var_name(name)
                js_code = f"""const {js_var_name} = builder.constant(
        {{dataType: '{webnn_dtype}', shape: [{dims_str}]}},
        new {typed_array}(weights_array_buffer, {offset}, {length} / {typed_array}.BYTES_PER_ELEMENT)
    );"""
            elif dims == ['1'] and data_type == 1 and "floatData" in initializer:
                js_var_name = to_js_var_name(name)
                value = initializer["floatData"][0]
                js_code = f"""const {js_var_name} = {value};"""
            else:
                js_code = f"""// Non-external initializer '{name}' is not handled."""
            js_lines.append("    " + js_code)

        # Generate all the operators
        op_handlers = {}

        # Handler for Conv -> WebNN conv2d
        def handle_conv(node):
            inputs = node.get("input", [])
            outputs = node.get("output", [])
            attrs = node.get("attribute", [])
            # Map ONNX attribute list to dict
            attr_dict = {a["name"]: a for a in attrs}
            input_vars = [to_js_var_name(i) for i in inputs]
            output_var = to_js_var_name(outputs[0])
            # Handle strides
            strides = attr_dict.get("strides", {}).get("ints", None)
            if strides is not None:
                assert len(strides) == 2, f"strides length must be 2, got {len(strides)}"
                strides_js = f"[{', '.join(str(s) for s in strides)}]"
            else:
                strides_js = "undefined"
            # Handle pads
            pads = attr_dict.get("pads", {}).get("ints", None)
            if pads is not None:
                assert len(pads) == 4, f"pads length must be 4, got {len(pads)}"
                pads_webnn = [pads[0], pads[2], pads[1], pads[3]]
                pads_js = f"[{', '.join(str(p) for p in pads_webnn)}]"
            else:
                pads_js = "undefined"
            # Handle dilations
            dilations = attr_dict.get("dilations", {}).get("ints", None)
            if dilations is not None:
                assert len(dilations) == 2, f"dilations length must be 2, got {len(dilations)}"
                dilations_js = f"[{', '.join(str(d) for d in dilations)}]"
            else:
                dilations_js = "undefined"
            groups = attr_dict.get("group", {}).get("i", None)
            js = f"""const {output_var} = builder.conv2d(
        {input_vars[0]}, {input_vars[1]},
        {{
            bias: {input_vars[2] if len(input_vars) > 2 else 'undefined'},
            strides: {strides_js},
            padding: {pads_js},
            dilations: {dilations_js},
            groups: {groups if groups is not None else 'undefined'}
        }}
    );"""
            return js

        # Handler for Clip -> WebNN clamp
        def handle_clip(node):
            inputs = node.get("input", [])
            outputs = node.get("output", [])
            input_vars = [to_js_var_name(i) for i in inputs]
            output_var = to_js_var_name(outputs[0])

            # ONNX Clip: input, min, max
            # WebNN clamp: builder.clamp(x, options)
            # options: {minValue, maxValue}
            min_value = input_vars[1] if len(input_vars) > 1 else 'undefined'
            max_value = input_vars[2] if len(input_vars) > 2 else 'undefined'

            js = f"""const {output_var} = builder.clamp(
        {input_vars[0]},
        {{
            minValue: {min_value},
            maxValue: {max_value}
        }}
    );"""
            return js

        # Handler for Add -> WebNN add
        def handle_add(node):
            inputs = node.get("input", [])
            outputs = node.get("output", [])
            input_vars = [to_js_var_name(i) for i in inputs]
            output_var = to_js_var_name(outputs[0])
            js = f"""const {output_var} = builder.add({input_vars[0]}, {input_vars[1]});"""
            return js

        # Handler for GlobalAveragePool -> WebNN averagePool2d
        def handle_globalaveragepool(node):
            inputs = node.get("input", [])
            outputs = node.get("output", [])
            input_vars = [to_js_var_name(i) for i in inputs]
            output_var = to_js_var_name(outputs[0])
            js = f"""const {output_var} = builder.averagePool2d(
        {input_vars[0]}
    );"""
            return js

        # Handler for Reshape -> WebNN reshape
        def handle_reshape(node):
            inputs = node.get("input", [])
            outputs = node.get("output", [])
            input_vars = [to_js_var_name(i) for i in inputs]
            output_var = to_js_var_name(outputs[0])

            # Assert inputs[1] is an initializer
            shape_name = inputs[1]
            assert shape_name in {init['name'] for init in initializers}, f"Reshape shape input '{shape_name}' must be an initializer"

            # Generate code to read shape from weights_array_buffer
            # Find the initializer entry
            shape_init = next(init for init in initializers if init['name'] == shape_name)
            assert "externalData" in shape_init, f"Reshape shape initializer '{shape_name}' must have externalData"
            shape_offset = None
            shape_length = None
            for entry in shape_init["externalData"]:
                if entry["key"] == "offset":
                    shape_offset = int(entry["value"])
                elif entry["key"] == "length":
                    shape_length = int(entry["value"])
            assert shape_offset is not None, f"Reshape shape initializer '{shape_name}' missing offset"
            assert shape_length is not None, f"Reshape shape initializer '{shape_name}' missing length"
            # Only support Int64Array for shape tensor
            js_shape_array = f"new BigInt64Array(weights_array_buffer, {shape_offset}, {shape_length} / BigInt64Array.BYTES_PER_ELEMENT)"
            # Convert BigInt64Array to Number array for WebNN and handle -1
            js_shape = (
                f"""(() => {{
        const shape = Array.from({js_shape_array}, Number);
        // Calculate the concrete size for value -1.
        if (shape.includes(-1)) {{
            const count = shape.filter(v => v === -1).length;
            if (count !== 1) {{
                throw new Error('Only one -1 is allowed in reshape shape');
            }}
            const totalInput = {input_vars[0]}.shape.reduce((a, b) => a * b, 1);
            const known = shape.reduce((a, b) => b === -1 ? a : a * b, 1);
            const idx = shape.indexOf(-1);
            shape[idx] = totalInput / known;
        }}
        return shape;
    }})()"""
            )

            js = f"""const {output_var} = builder.reshape(
        {input_vars[0]},
        {js_shape}
    );"""
            return js

        # Handler for Gemm -> WebNN gemm
        def handle_gemm(node):
            inputs = node.get("input", [])
            outputs = node.get("output", [])
            attrs = node.get("attribute", [])
            attr_dict = {a["name"]: a for a in attrs}
            input_vars = [to_js_var_name(i) for i in inputs]
            output_var = to_js_var_name(outputs[0])

            # Default values for alpha, beta, transA, transB
            alpha = attr_dict.get("alpha", {}).get("f", 1.0)
            beta = attr_dict.get("beta", {}).get("f", 1.0)
            transA = int(attr_dict.get("transA", {}).get("i", 0))
            transB = int(attr_dict.get("transB", {}).get("i", 0))

            # WebNN: builder.gemm(A, B, options)
            # If C is present, pass as 'C' in options
            options = [
                f"alpha: {alpha}",
                f"beta: {beta}",
                f"aTranspose: {str(bool(transA)).lower()}",
                f"bTranspose: {str(bool(transB)).lower()}"
            ]
            if len(input_vars) > 2:
                options.append(f"C: {input_vars[2]}")

            js = f"""const {output_var} = builder.gemm(
        {input_vars[0]},
        {input_vars[1]},
        {{
            {', '.join(options)}
        }}
    );"""
            return js

        # Register handlers
        op_handlers["Conv"] = handle_conv
        op_handlers["Clip"] = handle_clip
        op_handlers["Add"] = handle_add
        op_handlers["GlobalAveragePool"] = handle_globalaveragepool
        op_handlers["Reshape"] = handle_reshape
        op_handlers["Gemm"] = handle_gemm
        # Add more handlers as needed...

        # Generate operator JS code
        js_lines.append("    // Create graph operators.")
        if "graph" in onnx_json and "node" in onnx_json["graph"]:
            for node in onnx_json["graph"]["node"]:
                op_type = node.get("opType", "")
                handler = op_handlers.get(op_type)
                if handler:
                    js_code = handler(node)
                    js_lines.append("    " + js_code)
                else:
                    # Fallback: comment for unsupported ops
                    node_name = node.get("name", "")
                    js_lines.append(f"// Unsupported op: {op_type} (node: {node_name})")

        # After handling all nodes, build the WebNN graph with all outputs
        js_lines.append("    // Build graph with output operands.")
        graph_outputs = onnx_json['graph'].get('output', [])
        output_vars = [to_js_var_name(output['name']) for output in graph_outputs]
        output_names = [output['name'] for output in graph_outputs]
        if output_vars:
            if len(output_vars) == 1:
                js_lines.append(f"    this.graph_ = await builder.build({{'{output_names[0]}': {output_vars[0]}}});")
            else:
                outputs_map = ', '.join(f"'{name}': {var}" for name, var in zip(output_names, output_vars))
                js_lines.append(f"    this.graph_ = await builder.build({{{outputs_map}}});")

        # Create tensors for graph outputs.
        js_lines.append("    // Create graph output tensors.")
        for output_info in graph_outputs:
            name = output_info['name']
            dims = output_info.get('type', {}).get('tensorType', {}).get('shape', {}).get('dim', [])
            dims_str = ', '.join(str(d.get('dimValue', 1)) for d in dims)
            elem_type = output_info.get('type', {}).get('tensorType', {}).get('elemType', 1)
            webnn_dtype = webnn_type_map.get(elem_type, 'float32')
            js_code = f"""this.outputTensors_['{name}'] = await this.context_.createTensor(
        {{dataType: '{webnn_dtype}', shape: [{dims_str}], readable: true}}
    );"""
            js_lines.append("    " + js_code)

        js_lines.append("  }")  # end build

        # Generate run method.
        js_lines.append("  async run(inputs) {")
        js_lines.append("    // Set input buffers to input tensors using writeTensor (sync)")
        js_lines.append("    for (const name in inputs) {")
        js_lines.append("      if (!(name in this.inputTensors_)) throw new Error(`Unknown input: ${name}`);")
        js_lines.append("      this.context_.writeTensor(this.inputTensors_[name], inputs[name]);")
        js_lines.append("    }")
        js_lines.append("    // Compute the graph")
        js_lines.append("    await this.context_.dispatch(this.graph_, this.inputTensors_, this.outputTensors_);")
        js_lines.append("    // Read output tensors to buffers using readTensor (async)")
        js_lines.append("    const outputs = {};")
        js_lines.append("    for (const name in this.outputTensors_) {")
        js_lines.append("      const tensor = this.outputTensors_[name];")
        js_lines.append("      const buffer = await this.context_.readTensor(tensor);")
        js_lines.append("      let typedArrayCtor;")
        js_lines.append("      switch (tensor.dataType) {")
        js_lines.append("        case 'float32': typedArrayCtor = Float32Array; break;")
        js_lines.append("        case 'float64': typedArrayCtor = Float64Array; break;")
        js_lines.append("        case 'int32': typedArrayCtor = Int32Array; break;")
        js_lines.append("        case 'uint8': typedArrayCtor = Uint8Array; break;")
        js_lines.append("        case 'int8': typedArrayCtor = Int8Array; break;")
        js_lines.append("        case 'uint16': typedArrayCtor = Uint16Array; break;")
        js_lines.append("        case 'int16': typedArrayCtor = Int16Array; break;")
        js_lines.append("        case 'uint32': typedArrayCtor = Uint32Array; break;")
        js_lines.append("        case 'int64': typedArrayCtor = BigInt64Array; break;")
        js_lines.append("        case 'uint64': typedArrayCtor = BigUint64Array; break;")
        js_lines.append("        default: throw new Error(`Unhandled tensor dataType: ${tensor.dataType}`);")
        js_lines.append("      }")
        js_lines.append("      outputs[name] = new typedArrayCtor(buffer);")
        js_lines.append("    }")
        js_lines.append("    return outputs;")
        js_lines.append("  }")

        js_lines.append("}")    # end class

        # Write all generated JS code to model.js
        webnn_js_file_path = os.path.splitext(output_json_path)[0] + ".js"
        # Remove existing webnn js file before saving
        if os.path.exists(webnn_js_file_path):
            os.remove(webnn_js_file_path)
        with open(webnn_js_file_path, "w", encoding="utf-8") as f:
            f.write('\n\n'.join(js_lines))

        # Generate index.html to test the model
        html_path = os.path.join(os.path.dirname(output_json_path), "index.html")
        html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Test {class_name}</title>
</head>
<body>
    <h1>Test {webnn_js_file_path}</h1>
    <button id="run-btn">Build & Run Model</button>
    <label for="deviceType">Device:</label>
    <select id="deviceType">
        <option value="cpu">CPU</option>
        <option value="gpu" selected>GPU</option>
        <option value="npu">NPU</option>
    </select>
    <label for="numRuns">#Runs:</label>
    <input type="number" id="numRuns" value="1" min="1" style="width: 4em;">
    <pre id="output"></pre>
    <script type="module">
        import {{ {class_name} }} from './{os.path.basename(webnn_js_file_path)}';

        document.getElementById('run-btn').onclick = async () => {{
            const output = document.getElementById('output');
            output.textContent = 'Building model...\\n';
            try {{
                const deviceType = document.getElementById('deviceType').value || 'gpu';
                const t0 = performance.now();
                const model = new {class_name}();
                await model.build({{ deviceType: deviceType }});
                const t1 = performance.now();
                output.textContent += `Model built successfully. Build latency: ${{(t1 - t0).toFixed(2)}} ms\\n`;

                // Output input tensor info
                output.textContent += '\\nInput tensors:\\n';
                for (const name in model.inputTensors_) {{
                    const tensor = model.inputTensors_[name];
                    output.textContent += `  ${{name}}: shape=[${{tensor.shape}}], dataType=${{tensor.dataType}}\\n`;
                }}

                // Output output tensor info
                output.textContent += '\\nOutput tensors:\\n';
                for (const name in model.outputTensors_) {{
                    const tensor = model.outputTensors_[name];
                    output.textContent += `  ${{name}}: shape=[${{tensor.shape}}], dataType=${{tensor.dataType}}\\n`;
                }}
                output.textContent += '\\n';

                // Prepare dummy input data for testing (random values)
                const inputs = {{}};
                for (const name in model.inputTensors_) {{
                    const tensor = model.inputTensors_[name];
                    let TypedArrayCtor = Float32Array;
                    switch (tensor.dataType) {{
                        case 'float32': TypedArrayCtor = Float32Array; break;
                        case 'float64': TypedArrayCtor = Float64Array; break;
                        case 'int32': TypedArrayCtor = Int32Array; break;
                        case 'uint8': TypedArrayCtor = Uint8Array; break;
                        case 'int8': TypedArrayCtor = Int8Array; break;
                        case 'uint16': TypedArrayCtor = Uint16Array; break;
                        case 'int16': TypedArrayCtor = Int16Array; break;
                        case 'uint32': TypedArrayCtor = Uint32Array; break;
                        case 'int64': TypedArrayCtor = BigInt64Array; break;
                        case 'uint64': TypedArrayCtor = BigUint64Array; break;
                        default: throw new Error(`Unhandled input dataType: ${{tensor.dataType}}`);
                    }}
                    const size = tensor.shape.reduce((a, b) => a * b, 1);
                    const arr = new TypedArrayCtor(size);
                    // Fill with random values
                    if (TypedArrayCtor === Float32Array || TypedArrayCtor === Float64Array) {{
                        for (let i = 0; i < size; ++i) arr[i] = Math.random();
                    }} else if (TypedArrayCtor.BYTES_PER_ELEMENT === 8) {{
                        // BigInt64Array/BigUint64Array
                        for (let i = 0; i < size; ++i) arr[i] = BigInt(Math.floor(Math.random() * 100));
                    }} else {{
                        for (let i = 0; i < size; ++i) arr[i] = Math.floor(Math.random() * 100);
                    }}
                    inputs[name] = arr;
                }}

                output.textContent += 'Running inference...\\n';
                // Get number of runs from input
                let numRuns = parseInt(document.getElementById('numRuns').value) || 1;
                if (numRuns < 1) numRuns = 1;
                // Time model.run and print median inference latency
                const latencies = [];
                let results = null;
                for (let i = 0; i < numRuns; ++i) {{
                    const t0 = performance.now();
                    results = await model.run(inputs);
                    const t1 = performance.now();
                    latencies.push(t1 - t0);
                }}
                latencies.sort((a, b) => a - b);
                const median = latencies[Math.floor(latencies.length / 2)];
                output.textContent += `Median inference latency (${{numRuns}} runs): ${{median.toFixed(2)}} ms\\n`;
                output.textContent += '\\n';
                output.textContent += 'Inference results:\\n' + JSON.stringify(results, null, 2);
            }} catch (e) {{
                output.textContent += 'Error: ' + e;
            }}
        }};
    </script>
</body>
</html>
"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_code)

        return onnx_json


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-if',
        '--input_onnx_file_path',
        type=str,
        required=True,
        help='Input ONNX model path. (*.onnx)'
    )
    parser.add_argument(
        '-oj',
        '--output_json_path',
        type=str,
        required=True,
        help='Output JSON file path (*.json)'
    )
    parser.add_argument(
        '-i',
        '--json_indent',
        type=int,
        default=2,
        help='Number of indentations in JSON. (default=2)'
    )
    parser.add_argument(
        '-ew',
        '--external_weights',
        action='store_true',
        help='Store weights to an external file'
    )
    parser.add_argument(
        '-js',
        '--webnn_js',
        action='store_true',
        help='Generate WebNN JavaScript code, must be used together with --external_weights'
    )
    args = parser.parse_args()

    input_onnx_file_path = args.input_onnx_file_path
    output_json_path = args.output_json_path
    json_indent = args.json_indent
    external_weights = args.external_weights
    webnn_js = args.webnn_js

    # Convert onnx model to JSON
    onnx_graph = onnx.load(input_onnx_file_path)

    onnx_json = convert(
        input_onnx_file_path=None,
        onnx_graph=onnx_graph,
        output_json_path=output_json_path,
        json_indent=json_indent,
        external_weights=external_weights,
        webnn_js=webnn_js
    )


if __name__ == '__main__':
    main()
