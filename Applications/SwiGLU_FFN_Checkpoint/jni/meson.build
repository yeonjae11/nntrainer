myapp_sources= [
    'main.cpp',
    cifar_path / 'cifar_dataloader.cpp'
]

myapp_dependencies = [app_utils_dep,
  nntrainer_ccapi_dep
]

e = executable('nntrainer_swiglu_checkpoint',
  myapp_sources,
  include_directories: [include_directories('.'), cifar_include_dir],
  dependencies: myapp_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)
