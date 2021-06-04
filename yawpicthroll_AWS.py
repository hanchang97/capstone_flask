import boto3

#여기 변경해야함
# AWS_ACCESS_KEY_ID_REK = "AKIAZDET2MTHQUAZ3XNJ"
# AWS_SECRET_ACCESS_KEY_REK = "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"
# AWS_DEFAULT_REGION_REK = "ap-northeast-2"
#
# client_rekognition = boto3.client('rekognition', aws_access_key_id=AWS_ACCESS_KEY_ID_REK,
#                          aws_secret_access_key=AWS_SECRET_ACCESS_KEY_REK,
#                          region_name=AWS_DEFAULT_REGION_REK)


def defect_faces():
    client = boto3.client('rekognition')


    response = client.detect_faces(
        Image={
            'S3Object': {
                'Bucket': 'capstonefaceimg',
                'Name': 'user3TrainImage3.jpg',
            }
        }
    )

    print(response['FaceDetails'][0]['Pose'])
    return 'good'


