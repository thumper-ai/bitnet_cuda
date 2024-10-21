  # test.sh
  #!/bin/bash
  ctest --verbose
  cd pytorch_extension
  python -m pytest discover tests