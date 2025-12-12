myapp_sources = [
    'main.cpp'
]

myapp_dependencies = [
  app_utils_dep,
  nntrainer_ccapi_dep
]

e = executable('nntrainer_checkpoint_verification',
  myapp_sources,
  include_directories: include_directories('.'),
  dependencies: myapp_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)
