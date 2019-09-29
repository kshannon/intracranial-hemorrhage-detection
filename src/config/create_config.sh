cat > config.ini <<- "EOF"
    [path]
    s1_train_path = ''
    s1_test_path = ''
    docker_train = ../../data/stage_1_train_images/
    docker_test = ../../data/stage_1_test_images/
    train_csv_path = training.csv
    validate_csv_path = validation.csv
    test_csv_path = testing.csv
    [mode]
    use_docker = False
    gpu_rtx_20xx = False
EOF