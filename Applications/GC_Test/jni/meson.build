gelu_model_sources = ['gelu_model.cpp']
swiglu_model_sources = ['swiglu_model.cpp']

gelu_model_dependencies = [nntrainer_ccapi_dep]
swiglu_model_dependencies = [nntrainer_ccapi_dep]

gelu_gc_test_sources = [
  'gelu_model.cpp',
  'iteration_profiler.cpp'
]

swiglu_gc_test_sources = [
  'swiglu_model.cpp',
  'iteration_profiler.cpp'
]

e = executable('gelu_gc_test',
  gelu_gc_test_sources,
  dependencies: gelu_model_dependencies,
  include_directories: include_directories('.'),
  install: get_option('install-app'),
  install_dir: application_install_dir
)

e = executable('swiglu_gc_test',
  swiglu_gc_test_sources,
  dependencies: swiglu_model_dependencies,
  include_directories: include_directories('.'),
  install: get_option('install-app'),
  install_dir: application_install_dir
)
