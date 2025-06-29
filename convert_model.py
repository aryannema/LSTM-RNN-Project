import h5py, json
from keras.models import model_from_json

with h5py.File('next_word_lstm.h5', 'r') as f:
    model_config = f.attrs.get('model_config')
    config = json.loads(model_config)

    # Patch InputLayer and fix dtype fields
    for layer in config['config']['layers']:
        if layer['class_name'] == 'InputLayer' and 'batch_shape' in layer['config']:
            layer['config']['batch_input_shape'] = layer['config'].pop('batch_shape')
        if 'dtype' in layer['config'] and isinstance(layer['config']['dtype'], dict):
            layer['config']['dtype'] = layer['config']['dtype']['config']['name']

    # Recreate model
    model = model_from_json(json.dumps(config))

    # Load weights
    model.load_weights('next_word_lstm.h5')

model.save('next_word_lstm_tf', save_format='tf')

print("Model conversion and patch completed successfully.")
