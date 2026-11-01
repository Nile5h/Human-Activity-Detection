import cv2

WHITE_COLOR = (255, 255, 255)
GREEN_COLOR = (0, 255, 0)


def draw_line(image, p1, p2, color):
    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    cv2.line(image, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)


def draw_point(image, point):
    cv2.circle(
        image,
        (int(point[0]), int(point[1])),
        4,
        WHITE_COLOR,
        -1,
    )


def find_person_indicies(scores, score_threshold=0.9):
    return [i for i, s in enumerate(scores) if s > score_threshold]


def filter_persons(results, score_threshold=0.9):
    if not results:
        return ({}, [])

    result = results[0]
    if result.boxes is None or result.keypoints is None:
        return ({}, [])

    persons = {}
    scores = result.boxes.conf.cpu().numpy()
    p_indicies = find_person_indicies(scores, score_threshold)
    keypoints = result.keypoints.xy.cpu().numpy()
    for x in p_indicies:
        persons[x] = keypoints[x]
    return (persons, p_indicies)


def draw_keypoints(person, img):
    l_eye = person[1]
    r_eye = person[2]
    l_ear = person[3]
    r_ear = person[4]
    nose = person[0]
    l_shoulder = person[5]
    r_shoulder = person[6]
    l_elbow = person[7]
    r_elbow = person[8]
    l_wrist = person[9]
    r_wrist = person[10]
    l_hip = person[11]
    r_hip = person[12]
    l_knee = person[13]
    r_knee = person[14]
    l_ankle = person[15]
    r_ankle = person[16]

    draw_line(img, (l_shoulder[0], l_shoulder[1]),
              (l_elbow[0], l_elbow[1]), GREEN_COLOR)
    draw_line(img, (l_elbow[0], l_elbow[1]),
              (l_wrist[0], l_wrist[1]), GREEN_COLOR)
    draw_line(img, (l_shoulder[0], l_shoulder[1]),
              (r_shoulder[0], r_shoulder[1]), GREEN_COLOR)
    draw_line(img, (l_shoulder[0], l_shoulder[1]),
              (l_hip[0], l_hip[1]), GREEN_COLOR)
    draw_line(img, (r_shoulder[0], r_shoulder[1]),
              (r_hip[0], r_hip[1]), GREEN_COLOR)
    draw_line(img, (r_shoulder[0], r_shoulder[1]),
              (r_elbow[0], r_elbow[1]), GREEN_COLOR)
    draw_line(img, (r_elbow[0], r_elbow[1]),
              (r_wrist[0], r_wrist[1]), GREEN_COLOR)
    draw_line(img, (l_hip[0], l_hip[1]), (r_hip[0], r_hip[1]), GREEN_COLOR)
    draw_line(img, (l_hip[0], l_hip[1]), (l_knee[0], l_knee[1]), GREEN_COLOR)
    draw_line(img, (l_knee[0], l_knee[1]),
              (l_ankle[0], l_ankle[1]), GREEN_COLOR)
    draw_line(img, (r_hip[0], r_hip[1]), (r_knee[0], r_knee[1]), GREEN_COLOR)
    draw_line(img, (r_knee[0], r_knee[1]),
              (r_ankle[0], r_ankle[1]), GREEN_COLOR)

    for point in (
        l_eye, r_eye, l_wrist, r_wrist, l_shoulder, r_shoulder,
        l_elbow, r_elbow, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle,
    ):
        draw_point(img, point)
