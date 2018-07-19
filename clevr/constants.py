# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Store all the constants."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

QUESTIONS_PER_IMAGE = 10  # Name of questions per image
MAXSEQLENGTH = 50  # maximum length of a question
INPUTVOCABULARY = [
    'or', 'behind', 'that', 'red', 'shape', 'do', 'small', 'things', 'metal',
    'objects', 'block', 'blocks', 'visible', 'object', 'have', 'thing', 'side',
    'large', 'material', 'cubes', 'shiny', 'any', 'fewer', 'what', 'it', 'a',
    'made', 'more', 'spheres', 'gray', 'equal', 'the', 'other', 'color', 'big',
    'cyan', 'and', 'matte', 'balls', 'size', 'purple', 'in', 'its', 'another',
    'right', 'tiny', 'greater', 'anything', 'as', 'of', 'sphere', 'left', 'has',
    'cube', 'blue', 'on', 'either', 'there', 'yellow', 'how', 'to', 'same',
    'brown', 'does', 'less', 'many', 'cylinder', 'are', 'rubber', 'front', 'an',
    ';', 'than', 'green', 'metallic', '?', 'is', 'ball', 'number', 'cylinders',
    'both', 'else'
]

OUTPUTVOCABULARY = [
    'red', 'yellow', 'metal', 'no', '10', 'large', 'yes', '9', 'small', 'gray',
    'brown', 'cyan', 'purple', 'cube', 'blue', 'sphere', 'cylinder', 'rubber',
    'green', '8', '3', '2', '1', '0', '7', '6', '5', '4'
]


# If use popvec out_type
def get_prefs(grid_size):
  prefs_y, prefs_x = (np.mgrid[0:grid_size, 0:grid_size]) / (grid_size - 1.)
  prefs_x = prefs_x.flatten().astype('float32')
  prefs_y = prefs_y.flatten().astype('float32')

  # numpy array (Grid_size**2, 2)
  prefs = (np.array([prefs_x, prefs_y]).astype('float32')).T
  return prefs

GRID_SIZE = 7
PREFS = get_prefs(GRID_SIZE)

config = {'dataset': 'clevr',
          'pnt_net': False,
          'q_per_image': QUESTIONS_PER_IMAGE,
          'in_voc_size': len(INPUTVOCABULARY),
          'grid_size': GRID_SIZE,
          'out_voc_size': len(OUTPUTVOCABULARY),
          'maxseqlength': MAXSEQLENGTH,
          'prefs': PREFS,
         }
