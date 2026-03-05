pipeline {
    agent { label 'linux && gpu && compute' }

    environment {
        SONAR_HOST_URL    = 'http://127.0.0.1:9200'
        SONAR_PROJECT_KEY = 'diarized_transcriber'
        TRIVY_CACHE_DIR   = '/opt/trivy/cache'
        DEPLOY_PLAYBOOK   = 'deploy/ansible-deploy.yml'
    }

    options {
        timeout(time: 60, unit: 'MINUTES')
        disableConcurrentBuilds()
    }

    stages {
        stage('Trivy Security Scan') {
            steps {
                sh '''
                    /usr/bin/trivy fs \
                        --cache-dir "${TRIVY_CACHE_DIR}" \
                        --exit-code 1 \
                        --severity HIGH,CRITICAL \
                        --scanners vuln,secret \
                        --format table \
                        .
                '''
            }
        }

        stage('Run Tests') {
            steps {
                sh '/home/eddie/.local/bin/uv sync --extra dev --frozen'
                catchError(buildResult: 'UNSTABLE', stageResult: 'UNSTABLE') {
                    sh '''
                        .venv/bin/pytest tests/ \
                            --tb=short \
                            --ignore=tests/integration \
                            -q
                    '''
                }
            }
        }

        stage('SonarQube Analysis') {
            steps {
                withCredentials([string(credentialsId: 'sonarqube-token-diarized-transcriber', variable: 'SONAR_TOKEN')]) {
                    sh '''
                        sonar-scanner \
                            -Dsonar.projectKey="${SONAR_PROJECT_KEY}" \
                            -Dsonar.sources=src \
                            -Dsonar.tests=tests \
                            -Dsonar.python.coverage.reportPaths=coverage.xml \
                            -Dsonar.host.url="${SONAR_HOST_URL}" \
                            -Dsonar.token="${SONAR_TOKEN}"
                    '''
                }
            }
        }

        stage('Deploy') {
            steps {
                sh '''
                    # Run ansible with local connection (job already runs on tela)
                    ANSIBLE_HOST_KEY_CHECKING=False \
                    ansible-playbook \
                        -i "127.0.0.1," \
                        "${DEPLOY_PLAYBOOK}" \
                        --connection local \
                        -e "diarized_transcriber_src_dir=${WORKSPACE}"
                '''
            }
        }
    }

    post {
        always {
            deleteDir()
        }
        success {
            echo 'Pipeline completed successfully.'
        }
        failure {
            echo 'Pipeline failed.'
        }
    }
}
