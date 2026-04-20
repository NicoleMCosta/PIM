import cv2

#Este arquivo é para a criação do template que será usado para o template matching

img_rgb = cv2.imread('./frames/im_001.png', 0)
cantos = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cantos.append((x, y))


cv2.imshow('SELECIONE TOP LEFT E BOTTOM RIGHT', img_rgb)
cv2.setMouseCallback('SELECIONE TOP LEFT E BOTTOM RIGHT', mouse_callback)
cv2.waitKey(0)

if img_rgb is not None:
    x1, y1 = cantos[0]
    x2, y2 = cantos[1]
    
    start_x, end_x = min(x1, x2), max(x1, x2)
    start_y, end_y = min(y1, y2), max(y1, y2)
    template = img_rgb[start_y:end_y, start_x:end_x]

    if template.size > 0:
        cv2.imshow('Template Extraido', template)
        cv2.imwrite('./frames/template.png', template)
        cv2.waitKey(0)
        
cv2.destroyAllWindows()
