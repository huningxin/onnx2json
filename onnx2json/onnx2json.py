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
    weights_only: Optional[bool] = False,
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

    weights_only: Optional[bool]
        Save weights only.\n\
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

    if not onnx_graph:
        onnx_graph = onnx.load(input_onnx_file_path)

    if not weights_only:
        s = MessageToJson(onnx_graph)
        onnx_json = json.loads(s)

        if output_json_path:
            with open(output_json_path, 'w') as f:
                json.dump(onnx_json, f, indent=json_indent)
    else:
        external_data_file_path = os.path.splitext(output_json_path)[0] + ".bin"
        temp_external_data_onnx_file_path = "temp_external_data_model.onnx"
        onnx.save_model(onnx_graph, temp_external_data_onnx_file_path, save_as_external_data=True, all_tensors_to_one_file=True, location=external_data_file_path, size_threshold=0, convert_attribute=False)
        external_data_model = onnx.load(temp_external_data_onnx_file_path, load_external_data=False)
        os.remove(temp_external_data_onnx_file_path)
        s = MessageToJson(external_data_model)
        onnx_json = json.loads(s)
        initializer_josn = onnx_json['graph']['initializer']
        if output_json_path:
            with open(output_json_path, 'w') as f:
                json.dump(initializer_josn, f, indent=json_indent)

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
        '-wo',
        '--weights_only',
        action='store_true',
        help='Store weights only'
    )
    args = parser.parse_args()

    input_onnx_file_path = args.input_onnx_file_path
    output_json_path = args.output_json_path
    json_indent = args.json_indent
    weights_only = args.weights_only

    # Convert onnx model to JSON
    onnx_graph = onnx.load(input_onnx_file_path)

    onnx_json = convert(
        input_onnx_file_path=None,
        onnx_graph=onnx_graph,
        output_json_path=output_json_path,
        json_indent=json_indent,
        weights_only=weights_only
    )


if __name__ == '__main__':
    main()
