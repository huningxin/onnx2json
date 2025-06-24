#! /usr/bin/env python

import sys
import onnx
import json
import os
from google.protobuf.json_format import MessageToJson
from typing import Optional
from argparse import ArgumentParser
import base64
import numpy as np
import shutil

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
    output_js_path: Optional[str] = '',
    nhwc: Optional[bool] = False,
    dump_json: Optional[bool] = False,
    json_indent: Optional[int] = 2,
    imagenet: Optional[bool] = False
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

    output_js_path: Optional[str]
        Output WebNN JavaScript file path (*.js) If not specified, no JavaScript file is output.\n\
        Default: ''

    nhwc: Optional[bool]
        Generate WebNN operators taking nhwc input layout, including conv2d, convTranspose2d, resample2d and pool2d.\n\
        Default: false

    dump_json: Optional[bool]
        Dump the JSON representation of ONNX model.\n\
        Default: False

    json_indent: Optional[int]
        Number of indentations in JSON.\n\
        Default: 2

    imagenet: Optional[bool]
        Generate code in index.html for testing imagenet\n\
        Default: false

    Returns
    -------
    js_lines: str
        Converted WebNN JavaScript code.
    """

    # Unspecified check for input_onnx_file_path and onnx_graph
    if not input_onnx_file_path and not onnx_graph:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'One of input_onnx_file_path or onnx_graph must be specified.'
        )
        sys.exit(1)

    if not onnx_graph:
        onnx_graph = onnx.load(input_onnx_file_path)

    if not output_js_path:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'output_js_path must be specified.'
        )
        sys.exit(1)
    else:
        opset = onnx_graph.opset_import[0].version
        print(
            f'{Color.GREEN}INFO:{Color.RESET} '+
            f'The opset of the ONNX model is: {opset}.'
        )
        external_data_file_path = os.path.splitext(output_js_path)[0] + ".bin"
        # Remove existing external data file before saving
        if os.path.exists(external_data_file_path):
            os.remove(external_data_file_path)
        temp_external_data_onnx_file_path = "temp_external_data_model.onnx"
        onnx.save_model(onnx_graph, temp_external_data_onnx_file_path, save_as_external_data=True, all_tensors_to_one_file=True, location=external_data_file_path, size_threshold=0, convert_attribute=False)
        external_data_model = onnx.load(temp_external_data_onnx_file_path, load_external_data=False)
        os.remove(temp_external_data_onnx_file_path)
        s = MessageToJson(external_data_model)
        onnx_json = json.loads(s)

        if dump_json:
            output_json_path = os.path.splitext(output_js_path)[0] + ".json"
            with open(output_json_path, 'w') as f:
                json.dump(onnx_json, f, indent=json_indent)

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
        class_name = os.path.splitext(os.path.basename(output_js_path))[0]
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

        if os.path.exists(external_data_file_path):
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
""")
        js_lines.append(f"""
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
            9: 'Uint8Array',           # BOOL (special handling)
            10: 'Float16Array',  # FLOAT16 (not directly supported in JS)
            11: 'Float64Array',  # DOUBLE
            12: 'Uint32Array',   # UINT32
            13: 'BigUint64Array',   # UINT64 (not directly supported in JS)
        }

        # Helper to extract embedded data from an initializer as a Python array
        def get_initializer_embedded_data(initializer):
            """
            Returns the embedded data of an initializer as a Python array,
            handling all ONNX data types including float16.
            """
            data_type = initializer.get("dataType", 1)
            if data_type == 1:  # FLOAT
                return initializer.get("floatData", [])
            elif data_type == 2:  # UINT8
                return initializer.get("int32Data", [])
            elif data_type == 3:  # INT8
                return initializer.get("int32Data", [])
            elif data_type == 4:  # UINT16
                return initializer.get("uint16Data", [])
            elif data_type == 5:  # INT16
                return initializer.get("int16Data", [])
            elif data_type == 6:  # INT32
                return initializer.get("int32Data", [])
            elif data_type == 7:  # INT64
                return initializer.get("int64Data", [])
            elif data_type == 9:  # BOOL (stored as int32Data)
                return initializer.get("int32Data", [])
            elif data_type == 10:  # FLOAT16 (stored as uint16, convert to float16 then to float32 for JS)
                raw_data = initializer.get("float16Data", [])
                return np.array(raw_data, dtype=np.uint16).view(np.float16).astype(np.float32).tolist() if raw_data else []
            elif data_type == 11:  # DOUBLE
                return initializer.get("doubleData", [])
            elif data_type == 12:  # UINT32
                return initializer.get("uint32Data", [])
            elif data_type == 13:  # UINT64
                return initializer.get("uint64Data", [])
            else:
                return []

        def transpose_dims(dims, perm):
            """
            Transpose dims from NCHW to NHWC using permutation [0, 2, 3, 1].
            """
            if len(dims) == 4:
                return [dims[i] for i in perm]
            return dims

        initializers = onnx_json['graph'].get('initializer', [])

        # Generate all the graph input operands and tensors
        js_lines.append("    // Create graph input operands and tensors.")
        graph_inputs = onnx_json['graph'].get('input', [])
        initializer_names = {init['name'] for init in initializers}
        for input_info in graph_inputs:
            name = input_info['name']
            # Skip if input is also an initializer
            if name in initializer_names:
                continue
            dims = input_info.get('type', {}).get('tensorType', {}).get('shape', {}).get('dim', [])
            elem_type = input_info.get('type', {}).get('tensorType', {}).get('elemType', 1)
            webnn_dtype = webnn_type_map.get(elem_type, 'float32')
            js_var_name = to_js_var_name(name)
            if nhwc and len(dims) == 4:
                js_lines.append(f"""    // Input '{js_var_name}' layout is NHWC.""")
                dims = transpose_dims(dims, [0, 2, 3, 1])
            dims_str = ', '.join(str(d.get('dimValue', 1)) for d in dims)
            js_code = f"""const {js_var_name} = builder.input(
        '{name}',
        {{dataType: '{webnn_dtype}', shape: [{dims_str}]}}
    );"""
            js_lines.append("    " + js_code)
            js_lines.append(f"""    this.inputTensors_['{name}'] = await this.context_.createTensor(
        {{dataType: '{webnn_dtype}', shape: [{dims_str}], writable: true}}
    );""")

        # Record the DequantizeLinear nodes by their output name
        # It will be used by conv2d to find the real filter constant through dequantizeLinear op
        dequantizelinear_outputs = {}
        if "graph" in onnx_json and "node" in onnx_json["graph"]:
            for node in onnx_json["graph"]["node"]:
                if node.get("opType", "") == "DequantizeLinear":
                    outputs = node.get("output", [])
                    assert len(outputs) == 1, f"DequantizeLinear node must have exactly one output, got {len(outputs)}"
                    dequantizelinear_outputs[outputs[0]] = node

        # Helper to find the first input name of a DequantizeLinear node by its output name
        def find_dequantizelinear_input_by_output(output_name):
            """
            Given the output name of a DequantizeLinear node, return its first input name.
            Returns None if not found.
            """
            node = dequantizelinear_outputs.get(output_name)
            if node:
                inputs = node.get("input", [])
                if inputs:
                    return inputs[0]
            return None

        # Generate all the operators
        op_handlers = {}

        # Map ONNX data type to numpy dtype (shared for all helpers)
        onnx_dtype_to_np = {
            1: np.float32,   # FLOAT
            2: np.uint8,     # UINT8
            3: np.int8,      # INT8
            4: np.uint16,    # UINT16
            5: np.int16,     # INT16
            6: np.int32,     # INT32
            7: np.int64,     # INT64
            9: np.int32,     # BOOL (as int32)
            10: np.float16,  # FLOAT16
            11: np.float64,  # DOUBLE
            12: np.uint32,   # UINT32
            13: np.uint64,   # UINT64
        }

        constant_var_cache = {}
        # Helper to try to create a WebNN constant from an initializer name, with optional permutation, new_shape, and caching
        def try_create_constant(initializer_name, permutation=None, new_shape=None):
            """
            Given an initializer name, create the corresponding WebNN constant JS code if not created, otherwise return the cached JS variable name of the crated constant.
            If permutation is provided, transpose the data before creating the constant.
            If new_shape is provided, use it as the shape for the constant.
            Returns the JS variable name for the constant. If not an initializer, just returns the JS variable name.
            """
            cache_key = (initializer_name, tuple(permutation) if permutation is not None else None, tuple(new_shape) if new_shape is not None else None)
            js_var_name = to_js_var_name(initializer_name)
            if cache_key in constant_var_cache:
                return constant_var_cache[cache_key]

            initializer = next((init for init in initializers if init['name'] == initializer_name), None)
            if initializer is None:
                # Not an initializer, just return the JS variable name
                return js_var_name

            dims = list(new_shape) if new_shape is not None else initializer.get('dims', [])
            dims = [int(d) for d in dims]  # Ensure all dims are int
            if permutation != None:
                assert len(dims) == len(permutation), "Permutation must match filter dimensions"
            data_type = initializer.get('dataType', 1)
            webnn_dtype = webnn_type_map.get(data_type, 'float32')
            if permutation is not None:
                js_var_name += "_perm" + "".join(str(i) for i in permutation)
            if new_shape is not None:
                js_var_name += "_reshaped_" + "_".join(str(x) for x in new_shape)
            typed_array = typed_array_map.get(data_type, 'Float32Array')

            if "externalData" in initializer:
                # External data handling: load, transpose with numpy, and write back
                offset = 0
                length = 0
                location = None
                for entry in initializer["externalData"]:
                    if entry["key"] == "offset":
                        offset = int(entry["value"])
                    elif entry["key"] == "length":
                        length = int(entry["value"])
                    elif entry["key"] == "location":
                        location = entry["value"]
                # Read the external data file and transpose if needed
                if permutation is not None:
                    # Read, transpose, and write back, the weights_array_buffer will contain the transposed weights
                    np_dtype = onnx_dtype_to_np.get(data_type, np.float32)
                    with open(location, "r+b") as f:
                        f.seek(offset)
                        arr = np.frombuffer(f.read(length), dtype=np_dtype).reshape(tuple(dims))
                        arr_transposed = np.transpose(arr, permutation)
                        f.seek(offset)
                        f.write(arr_transposed.astype(np_dtype).tobytes())
                    
                    dims_perm = [dims[p] for p in permutation]
                    dims_str = ', '.join(str(d) for d in dims_perm)
                else:
                    dims_str = ', '.join(str(d) for d in dims)
                typed_array_bytes = f"{typed_array}.BYTES_PER_ELEMENT"
                base_buffer_var = f"{js_var_name}_buffer"
                js_code = (
                    f"""let {base_buffer_var} = new {typed_array}(weights_array_buffer, {offset}, {length} / {typed_array_bytes});
    const {js_var_name} = builder.constant(
        {{dataType: '{webnn_dtype}', shape: [{dims_str}]}},
        {base_buffer_var}
    );"""
                )
            else:
                # Embedded data
                py_data = get_initializer_embedded_data(initializer)
                if permutation is not None:
                    np_dtype = onnx_dtype_to_np.get(data_type, np.float32)
                    arr = np.array(py_data, dtype=np_dtype)
                    arr = arr.reshape(dims)
                    arr = np.transpose(arr, permutation)
                    arr = arr.flatten()
                    if webnn_dtype == "int64":
                        js_data = "[" + ", ".join(str(x) + "n" for x in arr) + "]"
                    else:
                        js_data = "[" + ", ".join(str(x) for x in arr) + "]"
                    dims_perm = [dims[p] for p in permutation]
                    dims_str = ', '.join(str(d) for d in dims_perm)
                else:
                    if webnn_dtype == "int64":
                        js_data = "[" + ", ".join(str(x) + "n" for x in py_data) + "]"
                    else:
                        js_data = "[" + ", ".join(str(x) for x in py_data) + "]"
                    dims_str = ', '.join(str(d) for d in dims)
                js_code = f"""const {js_var_name} = builder.constant(
        {{dataType: '{webnn_dtype}', shape: [{dims_str}]}},
        new {typed_array}({js_data})
    );"""
            js_lines.append("    " + js_code)
            constant_var_cache[cache_key] = js_var_name
            return js_var_name

        # Helper to fetch a tensor shape from valueInfo in the ONNX JSON.
        # ONNX Simplifier will do shape inference and add value info for each output tensor.
        def get_valueinfo_shape(name):
            """
            Returns the shape of a tensor given its name by searching valueInfo in the ONNX JSON.
            The shape is returned as a list of integers.
            """
            def extract_shape(info):
                tensor_type = info.get('type', {}).get('tensorType', None)
                assert tensor_type is not None, f"valueInfo for '{name}' is not a tensorType"
                dims = tensor_type.get('shape', {}).get('dim', [])
                return [int(d.get('dimValue', 1)) for d in dims]

            # Search valueInfo
            for vi in onnx_json['graph'].get('valueInfo', []):
                if vi['name'] == name:
                    return extract_shape(vi)
            # Search inputs
            for vi in onnx_json['graph'].get('input', []):
                if vi['name'] == name:
                    return extract_shape(vi)
            # Search outputs
            for vi in onnx_json['graph'].get('output', []):
                if vi['name'] == name:
                    return extract_shape(vi)
            return []

        # Helper to fetch a tensor shape by tensor name.
        # If the tensor is an initializer, get its shape from the initializer.
        # Otherwise, get its shape from valueInfo.
        def get_tensor_shape(name):
            """
            Returns the shape of a tensor given its name.
            If the tensor is an initializer, returns its 'dims'.
            Otherwise, uses get_valueinfo_shape to get the shape from valueInfo.
            """
            for init in initializers:
                if init['name'] == name:
                    return [int(d) for d in init.get('dims', [])]
            return get_valueinfo_shape(name)

        # Helper to fetch a tensor data type by tensor name.
        # If the tensor is an initializer, get its data type from the initializer.
        # Otherwise, get its data type from valueInfo.
        def get_tensor_dtype(name):
            """
            Returns the ONNX data type (int) of a tensor given its name.
            If the tensor is an initializer, returns its 'dataType'.
            Otherwise, uses valueInfo to get the elemType.
            """
            for init in initializers:
                if init['name'] == name:
                    return init.get('dataType', 1)
            for vi in onnx_json['graph'].get('valueInfo', []):
                if vi['name'] == name:
                    return vi.get('type', {}).get('tensorType', {}).get('elemType', 1)
            for vi in onnx_json['graph'].get('input', []):
                if vi['name'] == name:
                    return vi.get('type', {}).get('tensorType', {}).get('elemType', 1)
            for vi in onnx_json['graph'].get('output', []):
                if vi['name'] == name:
                    return vi.get('type', {}).get('tensorType', {}).get('elemType', 1)
            return 1  # Default to float32 if not found

        def handle_negative_axis(axis, input_rank):
            if axis < 0:
                axis += input_rank
            if axis < 0 or axis >= input_rank:
                raise ValueError(f"Axis {axis} is out of bounds for input rank {input_rank}.")
            return axis

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
            # Assert only support ONNX auto_pad is NOSET if present
            auto_pad = attr_dict.get("auto_pad", {}).get("s", None)
            if auto_pad is not None:
                # Decode base64 if needed
                try:
                    auto_pad_str = base64.b64decode(auto_pad).decode("utf-8")
                except Exception:
                    auto_pad_str = auto_pad
                assert auto_pad_str == "NOTSET", f"Only auto_pad=NOTSET is supported, got {auto_pad_str}"

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
            if groups is not None:
                groups = int(groups)

            # Check for depthwise conv2d: groups != 1 and groups == output_channel (filter_shape[0])
            is_depthwise = False
            filter_name = inputs[1]
            filter_var_name = input_vars[1]
            if groups is not None and groups != 1:
                filter_shape = get_tensor_shape(filter_name)
                assert len(filter_shape) == 4, f"Conv2d filter '{filter_name}' must have 4 dimensions, got {len(filter_shape)}"
                output_channels = filter_shape[0]
                if groups == int(output_channels):
                    is_depthwise = True

            # NHWC filter transpose: OIHW -> OHWI (normal) or OIHW -> IHWO (depthwise)
            filter_layout = None
            dq_input = None
            if not nhwc:
                filter_var_name = try_create_constant(filter_name)
            else:
                filter_name_for_transpose = None
                # If filter_name is not an initializer, try to find the DequantizeLinear input
                filter_is_initializer = any(init['name'] == filter_name for init in initializers)
                if not filter_is_initializer:
                    dq_input = find_dequantizelinear_input_by_output(filter_name)
                    assert dq_input is not None, f"Cannot find initializer or DequantizeLinear input for filter '{filter_name}'"
                    filter_name_for_transpose = dq_input
                else:
                    filter_name_for_transpose = filter_name

                if is_depthwise:
                    # Depthwise: OIHW -> IHWO
                    filter_var_name = try_create_constant(filter_name_for_transpose, (1, 2, 3, 0))
                    filter_layout = "'ihwo'"
                else:
                    # Regular: OIHW -> OHWI
                    filter_var_name = try_create_constant(filter_name_for_transpose, (0, 2, 3, 1))
                    filter_layout = "'ohwi'"

                # If filter_name was not an initializer, recreate WebNN dequantizeLinear op for the filter_var_name
                if dq_input is not None:
                    # Find the DequantizeLinear node for filter_name
                    dq_node = dequantizelinear_outputs.get(filter_name)
                    assert dq_node is not None, f"Cannot find DequantizeLinear node for filter '{filter_name}'"
                    # Replace the first input of the dq_node with filter_var_name
                    dq_node = dict(dq_node)  # Make a shallow copy to avoid mutating the original
                    dq_node["input"] = [filter_var_name] + dq_node["input"][1:]
                    dq_node["output"][0] = dq_node["output"][0] + "_transposed"
                    dq_js = op_handlers["DequantizeLinear"](dq_node)
                    js_lines.append(f"    {dq_js}")
                    filter_var_name = to_js_var_name(dq_node["output"][0])

            bias_var_name = 'undefined'
            if  len(inputs) > 2:
                bias_name = inputs[2]
                bias_var_name = try_create_constant(bias_name)

            options = [
                f"bias: {bias_var_name}",
                f"strides: {strides_js}",
                f"padding: {pads_js}",
                f"dilations: {dilations_js}",
                f"groups: {groups if groups is not None else 'undefined'}"
            ]
            if filter_layout:
                options.append(f"filterLayout: {filter_layout}")
            if nhwc:
                options.append("inputLayout: 'nhwc'")

            js = f"""const {output_var} = builder.conv2d(
        {input_vars[0]}, {filter_var_name},
        {{
            {', '.join(options)}
        }}
    );"""
            return js

        # Handler for ConvTranspose -> WebNN convTranspose2d
        def handle_convtranspose(node):
            inputs = node.get("input", [])
            outputs = node.get("output", [])
            attrs = node.get("attribute", [])
            attr_dict = {a["name"]: a for a in attrs}
            input_vars = [to_js_var_name(i) for i in inputs]
            output_var = to_js_var_name(outputs[0])

            # Assert only support ONNX auto_pad is NOTSET if present
            auto_pad = attr_dict.get("auto_pad", {}).get("s", None)
            if auto_pad is not None:
                # Decode base64 if needed
                try:
                    auto_pad_str = base64.b64decode(auto_pad).decode("utf-8")
                except Exception:
                    auto_pad_str = auto_pad
                assert auto_pad_str == "NOTSET", f"Only auto_pad=NOTSET is supported, got {auto_pad_str}"

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
            # Handle output_shape (optional) -> outputSizes in WebNN
            output_shape = attr_dict.get("output_shape", {}).get("ints", None)
            output_sizes_js = f"[{', '.join(str(s) for s in output_shape)}]" if output_shape is not None else "undefined"

            # NHWC filter transpose: IOHW -> OHWI
            filter_name = inputs[1]
            filter_var_name = input_vars[1]
            filter_layout = None
            if nhwc:
                filter_var_name = try_create_constant(filter_name, (1, 2, 3, 0))
                filter_layout = "'ohwi'"
            else:
                filter_var_name = try_create_constant(filter_name)

            bias_var_name = 'undefined'
            if  len(inputs) > 2:
                bias_name = inputs[2]
                bias_var_name = try_create_constant(bias_name)

            options = [
                f"bias: {bias_var_name}",
                f"strides: {strides_js}",
                f"padding: {pads_js}",
                f"dilations: {dilations_js}",
                f"groups: {groups if groups is not None else 'undefined'}",
                f"outputSizes: {output_sizes_js}"
            ]
            if filter_layout:
                options.append(f"filterLayout: {filter_layout}")
            if nhwc:
                options.append("inputLayout: 'nhwc'")

            js = f"""const {output_var} = builder.convTranspose2d(
        {input_vars[0]}, {filter_var_name},
        {{
            {', '.join(options)}
        }}
    );"""
            return js

        # Helper to extract array from initializer (externalData or embedded)
        def get_initializer_values(name, expected_dtype=None, expected_len=None):
            init = next((init for init in initializers if init['name'] == name), None)
            assert init is not None, f"'{name}' must be an initializer"
            data_type = init.get("dataType", 1)

            assert expected_dtype is not None, "expected_dtype must be specified"
            # Ensure expected_dtype can be a single type or a tuple of types
            if not isinstance(expected_dtype, (tuple, list)):
                expected_dtype = (expected_dtype,)
            if data_type not in expected_dtype:
                raise AssertionError(f"Initializer '{name}' must be dataType={expected_dtype}, got {data_type}")

            # External data
            if "externalData" in init:
                offset = None
                length = None
                for entry in init["externalData"]:
                    if entry["key"] == "offset":
                        offset = int(entry["value"])
                    elif entry["key"] == "length":
                        length = int(entry["value"])
                assert offset is not None and length is not None, f"Input '{name}' missing offset/length"
                np_dtype = onnx_dtype_to_np.get(data_type, np.float32)
                with open(external_data_file_path, "rb") as f:
                    f.seek(offset)
                    arr = np.frombuffer(f.read(length), dtype=np_dtype)
                    py = arr.tolist()
            else:
                # Embedded data
                py = get_initializer_embedded_data(init)
            if expected_len is not None:
                assert len(py) == expected_len, f"Expect {name} of length {expected_len}, got {len(py)}"
            return py
        
        # Helper to extract array from Constant node in the graph
        def get_constant_values(name, expected_dtype, expected_len=None):
            assert expected_dtype is not None, "expected_dtype must be specified"
            for n in onnx_json['graph'].get('node', []):
                if n.get('opType') == 'Constant' and name in n.get('output', []):
                    attrs = n.get('attribute', [])
                    for attr in attrs:
                        if attr.get("name") == "value" and "t" in attr:
                            value_attr = attr["t"]
                            arr = []
                            data_type = value_attr.get("dataType", None)
                            assert data_type == expected_dtype, f"Constant node '{name}' must have dataType={expected_dtype}, got {data_type}"
                            # Extract array based on data_type
                            if data_type == 1:  # FLOAT
                                arr = value_attr.get("floatData", [])
                            elif data_type == 2:  # UINT8
                                arr = value_attr.get("int32Data", [])
                            elif data_type == 3:  # INT8
                                arr = value_attr.get("int32Data", [])
                            elif data_type == 4:  # UINT16
                                arr = value_attr.get("uint16Data", [])
                            elif data_type == 5:  # INT16
                                arr = value_attr.get("int16Data", [])
                            elif data_type == 6:  # INT32
                                arr = value_attr.get("int32Data", [])
                            elif data_type == 7:  # INT64
                                arr = value_attr.get("int64Data", [])
                            elif data_type == 9:  # BOOL (stored as int32Data)
                                arr = value_attr.get("int32Data", [])
                            elif data_type == 10:  # FLOAT16 (stored as uint16, convert to float16 then to float32 for JS)
                                raw_data = value_attr.get("float16Data", [])
                                arr = np.array(raw_data, dtype=np.uint16).view(np.float16).astype(np.float32).tolist() if raw_data else []
                            elif data_type == 11:  # DOUBLE
                                arr = value_attr.get("doubleData", [])
                            elif data_type == 12:  # UINT32
                                arr = value_attr.get("uint32Data", [])
                            elif data_type == 13:  # UINT64
                                arr = value_attr.get("uint64Data", [])
                            else:
                                arr = []
                            if expected_len is not None:
                                assert len(arr) == expected_len, f"Constant node '{name}' must have length {expected_len}, got {len(arr)}"
                            return arr
                    break
            return None

        def get_tensor_values(name, expected_dtype, expected_len=None):
            """
            Returns the values for the given name.
            If the name is an initializer, calls get_initializer_values.
            Otherwise, calls get_constant_values and asserts result is not None.
            """
            if any(init['name'] == name for init in initializers):
                return get_initializer_values(name, expected_dtype, expected_len)
            else:
                values = get_constant_values(name, expected_dtype, expected_len)
                assert values is not None, f"{name} is neither an initializer nor a constant"
                return values

        # Handler for Clip -> WebNN clamp
        def handle_clip(node):
            inputs = node.get("input", [])
            outputs = node.get("output", [])
            input_vars = [to_js_var_name(i) for i in inputs]
            output_var = to_js_var_name(outputs[0])

            # ONNX Clip: input, min, max
            # WebNN clamp: builder.clamp(x, options)
            # options: {minValue, maxValue}
            min_value = 'undefined'
            max_value = 'undefined'

            if len(inputs) > 1 and inputs[1]:
                min_name = inputs[1]
                min_py = get_tensor_values(min_name, expected_dtype=1, expected_len=1)
                min_value = float(min_py[0])

            if len(inputs) > 2 and inputs[2]:
                max_name = inputs[2]
                max_py = get_tensor_values(max_name, expected_dtype=1, expected_len=1)
                max_value = float(max_py[0])

            js = f"""const {output_var} = builder.clamp(
        {input_vars[0]},
        {{
            minValue: {min_value},
            maxValue: {max_value}
        }}
    );"""
            return js

        # Handler for GlobalAveragePool -> WebNN averagePool2d
        def handle_globalaveragepool(node):
            inputs = node.get("input", [])
            outputs = node.get("output", [])
            input_vars = [to_js_var_name(i) for i in inputs]
            output_var = to_js_var_name(outputs[0])
            options = ""
            if nhwc:
                options = "{ layout: 'nhwc' }"
            js = f"""const {output_var} = builder.averagePool2d(
        {input_vars[0]}{', ' + options if options else ''}
    );"""
            return js

        # Generic handler for AveragePool and MaxPool -> WebNN averagePool2d / maxPool2d
        def make_pool_handler(webnn_op):
            def handler(node):
                inputs = node.get("input", [])
                outputs = node.get("output", [])
                attrs = node.get("attribute", [])
                attr_dict = {a["name"]: a for a in attrs}
                input_vars = [to_js_var_name(i) for i in inputs]
                output_var = to_js_var_name(outputs[0])

                # Handle strides
                strides = attr_dict.get("strides", {}).get("ints", [1, 1])
                assert len(strides) == 2, f"strides length must be 2, got {len(strides)}"
                strides_js = f"[{', '.join(str(s) for s in strides)}]"

                # Handle pads
                pads = attr_dict.get("pads", {}).get("ints", [0, 0, 0, 0])
                assert len(pads) == 4, f"pads length must be 4, got {len(pads)}"
                pads_webnn = [pads[0], pads[2], pads[1], pads[3]]
                pads_js = f"[{', '.join(str(p) for p in pads_webnn)}]"

                # Handle kernel_shape
                kernel_shape = attr_dict.get("kernel_shape", {}).get("ints", [0, 0])
                assert len(kernel_shape) == 2, f"kernel_shape length must be 2, got {len(kernel_shape)}"
                kernel_shape_js = f"[{', '.join(str(k) for k in kernel_shape)}]"

                # Handle dilations
                dilations = attr_dict.get("dilations", {}).get("ints", [1, 1])
                assert len(dilations) == 2, f"dilations length must be 2, got {len(dilations)}"
                dilations_js = f"[{', '.join(str(d) for d in dilations)}]"

                # Handle ceil_mode
                ceil_mode = int(attr_dict.get("ceil_mode", {}).get("i", 0))
                if ceil_mode not in (0, 1):
                    raise ValueError(f"ceil_mode must be 0 or 1, got {ceil_mode}")
                round_type_js = "'ceil'" if ceil_mode == 1 else "'floor'"

                # Handle count_include_pad (only for averagePool)
                count_include_pad = int(attr_dict.get("count_include_pad", {}).get("i", 0)) if webnn_op == "averagePool2d" else 0
                # WebNN doesn't support AveragePool with count_include_pad == 1, emulate it by pad + averagePool2d.
                if webnn_op == "averagePool2d" and count_include_pad == 1:
                    # Create pad options
                    input_shape = get_tensor_shape(inputs[0])
                    input_rank = len(input_shape)
                    beginning_padding = [0, 0, pads[0], pads[1]]
                    ending_padding = [0, 0, pads[2], pads[3]]
                    if nhwc:
                        beginning_padding = [0, pads[0], pads[1], 0]
                        ending_padding = [0, pads[2], pads[3], 0]

                    js_begin_padding_array = "[" + ", ".join(str(x) for x in beginning_padding) + "]"
                    js_end_padding_array = "[" + ", ".join(str(x) for x in ending_padding) + "]"

                    # Use pad op instead of padding in averagePool2d options
                    input_vars[0] = f"builder.pad({input_vars[0]}, {js_begin_padding_array}, {js_end_padding_array})"
                    # Unset padding option, because we will use pad op instead
                    pads_js = "undefined"  # No padding in averagePool2d options

                options = [
                    f"strides: {strides_js}",
                    f"padding: {pads_js}",
                    f"windowDimensions: {kernel_shape_js}",
                    f"dilations: {dilations_js}",
                    f"roundingType: {round_type_js}"
                ]
                if nhwc:
                    options.append("layout: 'nhwc'")
                js = f"""const {output_var} = builder.{webnn_op}(
        {input_vars[0]},
        {{
            {', '.join(options)}
        }}
    );"""
                return js
            return handler

        # Handler for Reshape -> WebNN reshape
        def handle_reshape(node):
            inputs = node.get("input", [])
            outputs = node.get("output", [])
            input_vars = [to_js_var_name(i) for i in inputs]
            output_var = to_js_var_name(outputs[0])

            # Assert inputs[1] is an initializer
            shape_name = inputs[1]
            assert shape_name in {init['name'] for init in initializers}, f"Reshape shape input '{shape_name}' must be an initializer"

            # Use get_tensor_values to get shape array as Python list
            shape_py = get_tensor_values(shape_name, expected_dtype=7)
            # Replace all 0 values in shape_py with 1
            shape_py = [1 if int(x) == 0 else int(x) for x in shape_py]
            # Convert shape array to JS array string
            js_shape_array = "[" + ", ".join(str(int(x)) for x in shape_py) + "]"
            # Handle -1 for dynamic shape
            js_shape = (
                f"""(() => {{
        const shape = {js_shape_array};
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

        # Handler for Resize -> WebNN resample2d
        def handle_resize(node):
            inputs = node.get("input", [])
            outputs = node.get("output", [])
            attrs = node.get("attribute", [])
            input_var = to_js_var_name(inputs[0])
            output_var = to_js_var_name(outputs[0])

            # Default mode is 'nearest'
            mode = "nearest"
            for attr in attrs:
                if attr.get("name") == "mode" and "s" in attr:
                    try:
                        mode_str = base64.b64decode(attr["s"]).decode("utf-8")
                    except Exception:
                        mode_str = attr["s"]
                    mode = mode_str.lower()

            # Map ONNX mode to WebNN mode
            if mode == "nearest":
                webnn_mode = "nearest-neighbor"
            elif mode == "linear":
                webnn_mode = "linear"
            elif mode == "cubic":
                raise AssertionError("WebNN does not support cubic mode for Resize.")
            else:
                webnn_mode = mode  # fallback, but should not happen

            scales_js = "undefined"
            sizes_js = "undefined"
            if len(inputs) > 3 and inputs[3]:
                sizes_name = inputs[3]
                sizes_py = get_tensor_values(sizes_name, expected_dtype=7, expected_len=4)
                sizes_js = f"[{int(sizes_py[2])}, {int(sizes_py[3])}]"
            if len(inputs) > 2 and inputs[2]:
                scales_name = inputs[2]
                scales_py = get_tensor_values(scales_name, expected_dtype=1, expected_len=4)
                scales_js = f"[{float(scales_py[2])}, {float(scales_py[3])}]"

            options = [
                f"mode: '{webnn_mode}'",
                f"scales: {scales_js}",
                f"sizes: {sizes_js}"
            ]
            if nhwc:
                options.append("axes: [1, 2]")

            js = f"""const {output_var} = builder.resample2d(
        {input_var},
        {{
            {', '.join(options)}
        }}
    );"""
            return js

        # Handler for Gemm -> WebNN gemm
        def handle_gemm(node):
            inputs = node.get("input", [])
            outputs = node.get("output", [])
            attrs = node.get("attribute", [])
            attr_dict = {a["name"]: a for a in attrs}
            a_var_name = to_js_var_name(inputs[0])
            # b might be a constant
            b_var_name = try_create_constant(inputs[1])
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
            if len(inputs) > 2:
                c_var_name = try_create_constant(inputs[2])
                options.append(f"C: {c_var_name}")

            js = f"""const {output_var} = builder.gemm(
        {a_var_name},
        {b_var_name},
        {{
            {', '.join(options)}
        }}
    );"""
            return js

        # Handler for Constant -> WebNN constant
        def handle_constant(node):
            outputs = node.get("output", [])
            output_var = to_js_var_name(outputs[0])
            attrs = node.get("attribute", [])
            value_attr = None
            for attr in attrs:
                if attr.get("name") == "value" and "t" in attr:
                    value_attr = attr["t"]
                    break
            if not value_attr:
                return f"// Constant node {outputs[0]} missing value tensor."

            data_type = value_attr.get("dataType", 1)
            webnn_dtype = webnn_type_map.get(data_type, "float32")
            dims = value_attr.get("dims", [])
            shape = [int(d) for d in dims] if dims else []

            # Get data and JS typed array
            if "floatData" in value_attr:
                data = value_attr["floatData"]
                js_data = "[" + ", ".join(str(x) for x in data) + "]"
                typed_array = "Float32Array"
            elif "float16Data" in value_attr:
                data = value_attr["float16Data"]
                js_data = "[" + ", ".join(str(x) for x in data) + "]"
                typed_array = "Float16Array"
            elif "int32Data" in value_attr:
                data = value_attr["int32Data"]
                js_data = "[" + ", ".join(str(x) for x in data) + "]"
                typed_array = "Int32Array"
            elif "int64Data" in value_attr:
                data = value_attr["int64Data"]
                js_data = "[" + ", ".join(str(x + "n") for x in data) + "]"
                typed_array = "BigInt64Array"
            elif "int8Data" in value_attr:
                data = value_attr["int8Data"]
                js_data = "[" + ", ".join(str(x) for x in data) + "]"
                typed_array = "Int8Array"
            elif "uint8Data" in value_attr:
                data = value_attr["uint8Data"]
                js_data = "[" + ", ".join(str(x) for x in data) + "]"
                typed_array = "Uint8Array"
            else:
                # fallback: fill with zeros
                size = 1
                for d in shape:
                    size *= d
                data = [0] * size
                js_data = "[" + ", ".join("0" for _ in range(size)) + "]"
                typed_array = "Float32Array"

            js = (f"""const {output_var} = builder.constant(
        {{dataType: '{webnn_dtype}', shape: {shape}}},
        new {typed_array}({js_data})
    );"""
                )
            return js

        # Handler for Transpose -> WebNN transpose
        def handle_transpose(node):
            inputs = node.get("input", [])
            outputs = node.get("output", [])
            attrs = node.get("attribute", [])
            input_var = to_js_var_name(inputs[0])
            output_var = to_js_var_name(outputs[0])
            # Default perm is reversed order if not specified
            perm = None
            for attr in attrs:
                if attr.get("name") == "perm" and "ints" in attr:
                    perm = attr["ints"]
                    break
            if perm is not None:
                js = f"""const {output_var} = builder.transpose(
        {input_var},
        {{ permutation: [{', '.join(str(p) for p in perm)}] }}
    );"""
            else:
                js = f"""const {output_var} = builder.transpose(
        {input_var}
    );"""
            return js

        # Handler for binary ops
        def make_binary_handler(op):
            def handler(node):
                inputs = node.get("input", [])
                outputs = node.get("output", [])
                a_var_name = try_create_constant(inputs[0])
                b_var_name = try_create_constant(inputs[1])
                output_var = to_js_var_name(outputs[0])
                js = f"""const {output_var} = builder.{op}({a_var_name}, {b_var_name});"""
                return js
            return handler

        # Handler for unary ops
        def make_unary_handler(op):
            def handler(node):
                inputs = node.get("input", [])
                outputs = node.get("output", [])
                input_var = to_js_var_name(inputs[0])
                output_var = to_js_var_name(outputs[0])
                js = f"""const {output_var} = builder.{op}({input_var});"""
                return js
            return handler

        # Handler for Softmax -> WebNN softmax
        def handle_softmax(node):
            inputs = node.get("input", [])
            outputs = node.get("output", [])
            attrs = node.get("attribute", [])
            input_var = to_js_var_name(inputs[0])
            output_var = to_js_var_name(outputs[0])

            # Default axis is 1 for ONNX Softmax
            axis = 1
            for attr in attrs:
                if attr.get("name") == "axis":
                    axis = int(attr.get("i", 1))
                    break

            js = f"""const {output_var} = builder.softmax(
        {input_var},
        {axis}
    );"""
            return js

        js_var_set = set()  # Track created JS variable names for reshaped scale and zero points.

        # Handler for DequantizeLinear and QuantizeLinear
        def make_qdq_handler(op):
            def handler(node):
                inputs = node.get("input", [])
                outputs = node.get("output", [])
                input_vars = [to_js_var_name(i) for i in inputs]
                output_var = to_js_var_name(outputs[0])

                # Get axis
                axis = 1  # ONNX default axis is 1
                for attr in node.get("attribute", []):
                    if attr.get("name") == "axis":
                        axis = int(attr.get("i", 1))
                        break

                # WebNN: builder.dequantizeLinear(x, scale, zeroPoint)
                # ONNX: input, scale, zero_point (optional)
                input_name = inputs[0]
                input_var = try_create_constant(input_name)

                input_shape = get_tensor_shape(inputs[0])
                scale_shape = get_tensor_shape(inputs[1])
                zero_point_shape = None
                zero_point_var = None
                if len(inputs) > 2:
                    zero_point_shape = get_tensor_shape(inputs[2])
                    zero_point_var = input_vars[2]
                
                # print(f"dequantizeLinear {outputs[0]} inputs {inputs[0]}: {input_shape}, {inputs[1]}: {scale_shape}, {inputs[2]}: {zero_point_shape}")

                # WebNN requires scale and zeroPoint have the same rank as input
                # If scale rank != input rank, expand scale shape for WebNN
                scale_name = inputs[1]
                scale_var = None
                if len(scale_shape) != len(input_shape):
                    # Assert scale is an initializer
                    assert any(init['name'] == scale_name for init in initializers), f"DequantizeLinear scale '{scale_name}' must be an initializer"
                    assert scale_shape == [] or len(scale_shape) == 1, f"DequantizeLinear scale shape must be scalar or 1D if not matching input rank, got {scale_shape}"
                    # Create new shape: all 1s, except axis
                    new_shape = [1] * len(input_shape)
                    if scale_shape and new_shape:  # not scalar
                        new_shape[axis] = scale_shape[0]
                    scale_var = try_create_constant(scale_name, None, new_shape)

                    # Reset the scale_shape, it could be used to create zero point if it
                    # is not present
                    scale_shape = new_shape

                    # Do the same for zero point if it is present
                    if len(inputs) == 3:
                        zero_point_name = inputs[2]
                        zero_point_var = try_create_constant(zero_point_name, None, new_shape)
                else:
                    scale_var = try_create_constant(scale_name)
                    if len(inputs) == 3:
                        zero_point_name = inputs[2]
                        zero_point_var = try_create_constant(zero_point_name)

                # Create a WebNN constant with value 0 and in shape of scale if zeroPoint is not present
                if len(inputs) == 2:
                    zero_point_shape = scale_shape
                    # Use get_tensor_dtype for zero_point dtype
                    zero_point_dtype_id = None
                    if op == "dequantizeLinear":
                        zero_point_dtype_id = get_tensor_dtype(inputs[0])
                    else:
                        assert op == "quantizeLinear", "Only support quantizeLinear or dequantizeLinear for QDQ handler"
                        zero_point_dtype_id = get_tensor_dtype(outputs[0])
                    zero_point_dtype = webnn_type_map.get(zero_point_dtype_id, 'float32')
                    zero_point_typed_array = typed_array_map.get(zero_point_dtype_id, 'Float32Array')
                    zero_point_js_var = to_js_var_name(input_vars[0]) + "_zero_point"
                    zero_point_data = [0] * (np.prod(zero_point_shape) if zero_point_shape else 1)
                    js_lines.append(f"""    const {zero_point_js_var} = builder.constant(
        {{dataType: '{zero_point_dtype}', shape: [{', '.join(str(x) for x in zero_point_shape)}]}},
        new {zero_point_typed_array}([{', '.join(str(x) for x in zero_point_data)}])
    );""")
                    zero_point_var = zero_point_js_var

                assert zero_point_var is not None

                js = f"""const {output_var} = builder.{op}(
        {input_var},
        {scale_var},
        {zero_point_var}
    );"""
                return js
            return handler

        # Handler for Pad -> WebNN pad
        def handle_pad(node):
            inputs = node.get("input", [])
            outputs = node.get("output", [])
            attrs = node.get("attribute", [])
            input_var = to_js_var_name(inputs[0])
            output_var = to_js_var_name(outputs[0])
            input_shape = get_tensor_shape(inputs[0])
            if input_shape is None:
                raise AssertionError(f"Cannot get shape of input tensor '{inputs[0]}'.")
            input_rank = len(input_shape)
            pads = None
            axes_py = None

            # Default mode is 'constant'
            mode = "constant"
            # Default value is 0.0
            value = 0.0
            for attr in attrs:
                if attr.get("name") == "mode" and "s" in attr:
                    try:
                        mode_str = base64.b64decode(attr["s"]).decode("utf-8")
                    except Exception:
                        mode_str = attr["s"]
                    mode = mode_str.lower()
                # Before Opset 11, constant value is in 'value' attribute
                # pads value is in 'pads' attribute
                elif attr.get("name") == "value" and "f" in attr:
                    value = attr["f"]
                elif attr.get("name") == "pads" and "ints" in attr: # Pad - 2
                    pads = attr["ints"]
                elif attr.get("name") == "paddings" and "ints" in attr: # Pad - 1
                    pads = attr["ints"]


            # Map ONNX mode to WebNN mode
            if mode == "constant":
                webnn_mode = "constant"
            elif mode == "reflect":
                webnn_mode = "reflection"
            elif mode == "edge":
                webnn_mode = "edge"
            else:
                raise AssertionError(f"WebNN does not support {mode} mode for Pad.")

            # Since Opset 11, pads and constant_value are inputs, and must be constants
            if opset >= 11:
                # pads is provided as constant
                if len(inputs) > 1 and inputs[1]:
                    pads_name = inputs[1]
                    pads = get_tensor_values(pads_name, expected_dtype=7)
                # constant_value is provided as consant
                if len(inputs) > 2 and inputs[2]:
                    value_name = inputs[2]
                    value_py = get_tensor_values(value_name, expected_dtype=1, expected_len=1)
                    value = float(value_py[0])
                # axes is provided as constant
                if len(inputs) > 3 and inputs[3]:
                    axes_name = inputs[3]
                    axes_py = get_tensor_values(axes_name, expected_dtype=[6, 7])
                    # Handle negative axis
                    axes_py = [handle_negative_axis(int(axis), input_rank) for axis in axes_py]

            beginning_padding = [0] * input_rank
            ending_padding = [0] * input_rank
            if axes_py is not None:
                # If axes is provided, use it to determine beginning and ending padding
                for i in range(len(axes_py)):
                    index = axes_py[i]
                    beginning_padding[index] = pads[i]
                    ending_padding[index] = pads[i + len(pads) // 2]
            else:
                beginning_padding = pads[:len(pads) // 2]
                ending_padding = pads[len(pads) // 2:]

            # Clamp negative pads to 0
            beginning_padding = [max(0, int(pad)) for pad in beginning_padding]
            ending_padding = [max(0, int(pad)) for pad in ending_padding]
            if nhwc:
                # For NHWC, we need to adjust the padding order
                if input_rank != 4:
                    # Currently NHWC handler is only supported for rank 4 inputs
                    raise AssertionError(f"NHWC padding requires rank 4 input, got {input_rank}.")
                beginning_padding = [beginning_padding[0]] + beginning_padding[2:4] + [beginning_padding[1]]
                ending_padding = [ending_padding[0]] + ending_padding[2:4] + [ending_padding[1]]
            # Convert beginning_padding and ending_padding arrays to JS array string
            js_begin_padding_array = "[" + ", ".join(str(int(x)) for x in beginning_padding) + "]"
            js_end_padding_array = "[" + ", ".join(str(int(x)) for x in ending_padding) + "]"

            options = [
                f"mode: '{webnn_mode}'",
                f"value: {value}",
            ]
            # WebNN pad: builder.pad(x, beginningPadding, endingPadding, options)
            js = f"""const {output_var} = builder.pad(
        {input_var}, {js_begin_padding_array}, {js_end_padding_array},
        {{
            {', '.join(options)}
        }}
    );"""
            return js

        # Handler for HardSigmoid -> WebNN hardSigmoid
        def handle_hardsigmoid(node):
            inputs = node.get("input", [])
            outputs = node.get("output", [])
            attrs = node.get("attribute", [])
            input_var = to_js_var_name(inputs[0])
            output_var = to_js_var_name(outputs[0])

            # Default values for alpha and beta
            alpha = 0.2
            beta = 0.5
            for attr in attrs:
                if attr.get("name") == "alpha":
                    alpha = float(attr.get("f", 0.2))
                elif attr.get("name") == "beta":
                    beta = float(attr.get("f", 0.5))

            js = f"""const {output_var} = builder.hardSigmoid(
        {input_var},
        {{ alpha: {alpha}, beta: {beta} }}
    );"""
            return js

        # Handler for Flatten -> WebNN reshape
        def handle_flatten(node):
            inputs = node.get("input", [])
            outputs = node.get("output", [])
            attrs = node.get("attribute", [])
            input_var = to_js_var_name(inputs[0])
            output_var = to_js_var_name(outputs[0])

            # Default axis is 1
            axis = 1
            for attr in attrs:
                if attr.get("name") == "axis":
                    axis = int(attr.get("i", 1))
                    break

            input_shape = get_tensor_shape(inputs[0])
            input_rank = len(input_shape)
            # Handle negative axis
            axis = handle_negative_axis(axis, input_rank)

            # Compute new shape: [dim0*...*dim(axis-1), dim(axis)*...*dimN]
            dim0 = 1
            for i in range(axis):
                dim0 *= input_shape[i]
            dim1 = 1
            for i in range(axis, input_rank):
                dim1 *= input_shape[i]
            new_shape = [dim0, dim1]

            js = f"""const {output_var} = builder.reshape(
        {input_var},
        [{new_shape[0]}, {new_shape[1]}]
    );"""
            return js

        # Handler for Concat -> WebNN concat
        def handle_concat(node):
            inputs = node.get("input", [])
            outputs = node.get("output", [])
            attrs = node.get("attribute", [])
            input_vars = [to_js_var_name(i) for i in inputs]
            output_var = to_js_var_name(outputs[0])
            input_shape = get_tensor_shape(inputs[0])
            if input_shape is None:
                raise AssertionError(f"Cannot get shape of input tensor '{inputs[0]}'.")

            # Default axis is 1
            axis = 1
            for attr in attrs:
                if attr.get("name") == "axis":
                    axis = int(attr.get("i", 1))
                    break
            axis = handle_negative_axis(axis, len(input_shape))

            # Adjust axis for NHWC layout if needed
            axis_comment = ""
            if nhwc and len(input_shape) == 4:
                # NCHW: 0-N, 1-C, 2-H, 3-W -> NHWC: 0-N, 1-H, 2-W, 3-C
                orig_axis = axis
                if axis == 1:
                    axis = 3  # channel axis moves to last
                elif axis == 2:
                    axis = 1  # height axis moves to 1
                elif axis == 3:
                    axis = 2  # width axis moves to 2
                if orig_axis != axis:
                    axis_comment = f" // axis permuted from {orig_axis} to {axis} for NHWC"

            js = f"""const {output_var} = builder.concat(
        [{', '.join(input_vars)}],
        {axis},{axis_comment}
    );"""
            return js

        # Register handlers
        op_handlers["Add"] = make_binary_handler("add")
        op_handlers["AveragePool"] = make_pool_handler("averagePool2d")
        op_handlers["Clip"] = handle_clip
        op_handlers["Concat"] = handle_concat
        op_handlers["Constant"] = handle_constant
        op_handlers["Conv"] = handle_conv
        op_handlers["ConvTranspose"] = handle_convtranspose
        op_handlers["DequantizeLinear"] = make_qdq_handler("dequantizeLinear")
        op_handlers["Div"] = make_binary_handler("div")
        op_handlers["Dropout"] = make_unary_handler("identity")
        op_handlers["Flatten"] = handle_flatten
        op_handlers["Gemm"] = handle_gemm
        op_handlers["GlobalAveragePool"] = handle_globalaveragepool
        op_handlers["HardSigmoid"] = handle_hardsigmoid
        op_handlers["HardSwish"] = make_unary_handler("hardSwish")
        op_handlers["MatMul"] = make_binary_handler("matmul")
        op_handlers["MaxPool"] = make_pool_handler("maxPool2d")
        op_handlers["Mul"] = make_binary_handler("mul")
        op_handlers["QuantizeLinear"] = make_qdq_handler("quantizeLinear")
        op_handlers["Pad"] = handle_pad
        op_handlers["Relu"] = make_unary_handler("relu")
        op_handlers["Reshape"] = handle_reshape
        op_handlers["Resize"] = handle_resize
        op_handlers["Sigmoid"] = make_unary_handler("sigmoid")
        op_handlers["Softmax"] = handle_softmax
        op_handlers["Sub"] = make_binary_handler("sub")
        op_handlers["Transpose"] = handle_transpose
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
                    node_name = node.get("name", "")
                    raise RuntimeError(f"Unsupported op: {op_type} (node: {node_name})")

        # After handling all nodes, build the WebNN graph with all outputs
        js_lines.append("    // Build graph with output operands.")
        graph_outputs = onnx_json['graph'].get('output', [])
        output_vars = [to_js_var_name(output['name']) for output in graph_outputs]
        output_names = [output['name'] for output in graph_outputs]

        # If nhwc, insert a transpose from NHWC to NCHW for each output, and replace output_var only after generating the transpose code
        if nhwc:
            for i, (output_var, output_info) in enumerate(zip(output_vars, graph_outputs)):
                dims = output_info.get('type', {}).get('tensorType', {}).get('shape', {}).get('dim', [])
                if len(dims) == 4:
                    js_lines.append(f"""    // Output '{output_var}' layout is NHWC.""")

        if output_vars:
            if len(output_vars) == 1:
                js_lines.append(f"    this.graph_ = await builder.build({{'{output_names[0]}': {output_vars[0]}}});")
            else:
                outputs_map = ', '.join(f"'{name}': {var}" for name, var in zip(output_names, output_vars))
                js_lines.append(f"    this.graph_ = await builder.build({{{outputs_map}}});")

        # Create tensors for graph outputs.
        js_lines.append("    // Create graph output tensors.")
        for i, (output_var, output_info) in enumerate(zip(output_vars, graph_outputs)):
            name = output_info['name']
            output_var = output_vars[i]
            js_code = f"""this.outputTensors_['{name}'] = await this.context_.createTensor(
        {{dataType: {output_var}.dataType, shape: {output_var}.shape, readable: true}}
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
        js_lines.append("        case 'float16': typedArrayCtor = Float16Array; break;")
        js_lines.append("        case 'int32': typedArrayCtor = Int32Array; break;")
        js_lines.append("        case 'uint8': typedArrayCtor = Uint8Array; break;")
        js_lines.append("        case 'int8': typedArrayCtor = Int8Array; break;")
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

        # Write all generated JS code to output_js_path
        # Remove existing webnn js file before saving
        if os.path.exists(output_js_path):
            os.remove(output_js_path)
        with open(output_js_path, "w", encoding="utf-8") as f:
            f.write('\n\n'.join(js_lines))

        if imagenet:
            # Try to copy labels1000.txt from the ONNX model's folder to the JS output folder
            onnx_dir = os.path.dirname(input_onnx_file_path)
            js_dir = os.path.dirname(output_js_path)
            src_labels = os.path.join(onnx_dir, "labels1000.txt")
            dst_labels = os.path.join(js_dir, "labels1000.txt")
            if os.path.isfile(src_labels):
                try:
                    shutil.copyfile(src_labels, dst_labels)
                    print(
                        f'{Color.GREEN}INFO:{Color.RESET} '+
                        f"Copied {src_labels} to {dst_labels}"
                    )
                except Exception as e:
                    print(f"Warning: Could not copy labels1000.txt: {e}")
            else:
                print(f"Warning: labels1000.txt not found in {onnx_dir}")

        # Generate index.html to test the model
        html_path = os.path.join(os.path.dirname(output_js_path), "index.html")
        html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Test {class_name}</title>
</head>
<body>
    <h1>Test {output_js_path}</h1>
    {('<div id="imgblock"><input type="file" id="imgfile" accept="image/*"><canvas id="imgcanvas" style="display:none"></canvas></div>') if imagenet else ""}
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
    <div id="layout-info" style="margin-top:1em;color:#444;"></div>
    <script type="module">
        import {{ {class_name} }} from './{os.path.basename(output_js_path)}';

        {"// Convert image file to Float32Array in NCHW or NHWC and display image\n"
         f"async function getImageInputData(inputShape, nhwc = {'true' if nhwc else 'false'}) {{\n"
         "    return new Promise((resolve, reject) => {\n"
         "        const fileInput = document.getElementById('imgfile');\n"
         "        if (!fileInput.files.length) {\n"
         "            reject('No image selected.');\n"
         "            return;\n"
         "        }\n"
         "        const file = fileInput.files[0];\n"
         "        const reader = new FileReader();\n"
         "        reader.onload = function(e) {\n"
         "            const img = new Image();\n"
         "            img.onload = function() {\n"
         "                let n, c, h, w;\n"
         "                if (inputShape.length === 4) {\n"
         "                    if (nhwc) {\n"
         "                        [n, h, w, c] = inputShape;\n"
         "                    } else {\n"
         "                        [n, c, h, w] = inputShape;\n"
         "                    }\n"
         "                } else {\n"
         "                    [n, c, h, w] = [1, 3, 224, 224];\n"
         "                }\n"
         "                const canvas = document.getElementById('imgcanvas');\n"
         "                canvas.width = w;\n"
         "                canvas.height = h;\n"
         "                const ctx = canvas.getContext('2d');\n"
         "                ctx.drawImage(img, 0, 0, w, h);\n"
         "                canvas.style.display = '';\n"
         "                const imgData = ctx.getImageData(0, 0, w, h).data;\n"
         "                let arr = new Float32Array(n * c * h * w);\n"
         "                const mean = [0.485, 0.456, 0.406];\n"
         "                const std = [0.229, 0.224, 0.225];\n"
         "                if (nhwc) {\n"
         "                    for (let y = 0; y < h; ++y) {\n"
         "                        for (let x = 0; x < w; ++x) {\n"
         "                            for (let ch = 0; ch < c; ++ch) {\n"
         "                                let v = imgData[(y * w + x) * 4 + ch] / 255.0;\n"
         "                                v = (v - mean[ch]) / std[ch];\n"
         "                                arr[y * w * c + x * c + ch] = v;\n"
         "                            }\n"
         "                        }\n"
         "                    }\n"
         "                } else {\n"
         "                    for (let ch = 0; ch < c; ++ch) {\n"
         "                        for (let y = 0; y < h; ++y) {\n"
         "                            for (let x = 0; x < w; ++x) {\n"
         "                                let v = imgData[(y * w + x) * 4 + ch] / 255.0;\n"
         "                                v = (v - mean[ch]) / std[ch];\n"
         "                                arr[ch * h * w + y * w + x] = v;\n"
         "                            }\n"
         "                        }\n"
         "                    }\n"
         "                }\n"
         "                resolve(arr);\n"
         "            };\n"
         "            img.onerror = reject;\n"
         "            img.src = e.target.result;\n"
         "        };\n"
         "        reader.onerror = reject;\n"
         "        reader.readAsDataURL(file);\n"
         "    });\n"
         "}\n"
         if imagenet else ""}

        async function getInputs(model) {{
            const inputs = {{}};
            for (const name in model.inputTensors_) {{
                const tensor = model.inputTensors_[name];
                if ({'true' if imagenet else 'false'}) {{
                    // Use image input
                    inputs[name] = await getImageInputData(tensor.shape);
                }} else {{
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
                        for (let i = 0; i < size; ++i) arr[i] = BigInt(Math.floor(Math.random() * 100));
                    }} else {{
                        for (let i = 0; i < size; ++i) arr[i] = Math.floor(Math.random() * 100);
                    }}
                    inputs[name] = arr;
                }}
            }}
            return inputs;
        }}

        async function showTop5(results) {{
            // Fetch labels if not already loaded
            if (!window.imagenetLabels) {{
                const resp = await fetch('labels1000.txt');
                window.imagenetLabels = (await resp.text()).split('\\n').map(s => s.trim()).filter(s => s.length > 0);
            }}
            const labels = window.imagenetLabels;
            // Assume single output
            const output = Object.values(results)[0];
            let arr = Array.from(output);
            // Always apply softmax
            function softmax(arr) {{
                const max = Math.max(...arr);
                const exps = arr.map(x => Math.exp(x - max));
                const sum = exps.reduce((a, b) => a + b, 0);
                return exps.map(e => e / sum);
            }}
            arr = softmax(arr);
            let top5 = arr.map((v, i) => [v, i])
                .sort((a, b) => b[0] - a[0])
                .slice(0, 5);
            let msg = "Top 5 results:\\n";
            for (const [score, idx] of top5) {{
                const label = labels && labels[idx] ? labels[idx] : `#${{idx}}`;
                msg += `  ${{label}}: ${{(score * 100).toFixed(2)}}%\\n`;
            }}
            document.getElementById('output').textContent += msg;
        }}

        async function runModel() {{
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
                {"if (!" + str(imagenet).lower() + ") {" if imagenet else ""}
                output.textContent += '\\nInput tensors:\\n';
                for (const name in model.inputTensors_) {{
                    const tensor = model.inputTensors_[name];
                    output.textContent += `  ${{name}}: shape=[${{tensor.shape}}], dataType=${{tensor.dataType}}\\n`;
                }}
                output.textContent += '\\n';

                // Output output tensor info
                output.textContent += '\\nOutput tensors:\\n';
                for (const name in model.outputTensors_) {{
                    const tensor = model.outputTensors_[name];
                    output.textContent += `  ${{name}}: shape=[${{tensor.shape}}], dataType=${{tensor.dataType}}\\n`;
                }}
                output.textContent += '\\n';
                {"}" if imagenet else ""}

                // Prepare input data
                const inputs = await getInputs(model);

                output.textContent += 'Running inference...\\n';
                let numRuns = parseInt(document.getElementById('numRuns').value) || 1;
                if (numRuns < 1) numRuns = 1;
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
                {"await showTop5(results);" if imagenet else "output.textContent += 'Inference results:\\n' + JSON.stringify(results, null, 2) + '\\n';"}

            }} catch (e) {{
                output.textContent += 'Error: ' + e;
            }}
        }}

        window.addEventListener('DOMContentLoaded', () => {{
            document.getElementById('run-btn').onclick = runModel;
        }});
    </script>
</body>
</html>
"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_code)

        return js_lines


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
        '--output_js_path',
        type=str,
        required=True,
        help='Output WebNN JavaScript file path (*.js)'
    )
    parser.add_argument(
        '-nhwc',
        '--nhwc',
        action='store_true',
        help='Generate WebNN operators taking nhwc input layout, including conv2d, convTranspose2d, resample2d and pool2d'
    )
    parser.add_argument(
        '-json',
        '--dump_json',
        action='store_true',
        help='Dump the JSON representation of ONNX model'
    )
    parser.add_argument(
        '-i',
        '--json_indent',
        type=int,
        default=2,
        help='Number of indentations in JSON. (default=2)'
    )
    parser.add_argument(
        '-imagenet',
        '--imagenet',
        action='store_true',
        help='Test imagenet model in the generated index.html'
    )

    args = parser.parse_args()

    input_onnx_file_path = args.input_onnx_file_path
    output_js_path = args.output_js_path
    nhwc = args.nhwc
    dump_json = args.dump_json
    json_indent = args.json_indent
    imagenet = args.imagenet

    onnx_json = convert(
        input_onnx_file_path=input_onnx_file_path,
        onnx_graph=None,
        output_js_path=output_js_path,
        nhwc=nhwc,
        dump_json=dump_json,
        json_indent=json_indent,
        imagenet=imagenet
    )


if __name__ == '__main__':
    main()
