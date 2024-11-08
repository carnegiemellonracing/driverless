# TODO Add a scaling system so that each grid block is 1m across
import pygame
import sys
import os
from datetime import datetime
import math

pygame.init()

WIDTH, HEIGHT = 800, 600
BLACK = (0,0,0)
WHITE = (255, 255, 255)
GREY = (45, 45, 45)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

class TrackVisualizer:
  def __init__(self):
    self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
    self.origin_x = WIDTH // 2
    self.origin_y = HEIGHT - 50
    self.click_points = []
    self.scale = 20
    self.grid_size = self.scale
    self.offset_x = 0
    self.offset_y = 0
    self.adjusted_origin_x = self.origin_x + self.offset_x
    self.adjusted_origin_y = self.origin_y + self.offset_y
        
    if not os.path.exists("data"):
      os.makedirs("data")
      
  def draw_grid(self):
    for x in range(0, WIDTH, self.grid_size):
      for y in range(0, HEIGHT, self.grid_size):
        rect = pygame.Rect(x, y, self.grid_size, self.grid_size)
        pygame.draw.rect(self.screen, GREY, rect, 1)
  
  def add_point(self, x, y, button):
    adjusted_x = x - self.offset_x
    adjusted_y = y - self.offset_y
    if button == 1: side = "left"
    elif button == 3: side = "right"
    else: side = "ERROR"
    self.click_points.append((adjusted_x, adjusted_y, side))
  
  def save_points(self):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_points = f"data/track_points_{timestamp}.txt"
    filename_seg = f"data/track_segs_{timestamp}.txt"
    
    with open(filename_points, 'w') as file:
      for point in self.click_points:
        x_meters = point[0] / self.scale
        y_meters = point[1] / self.scale
        file.write(f"{x_meters},{y_meters},{point[2]}\n")
      print(f"Points saved to {filename_points}\n")
      
  def pan(self, delta_x, delta_y):
    self.offset_x += delta_x
    self.offset_y += delta_y
      
  def draw(self):
    self.screen.fill(BLACK)
    self.draw_grid()
    
    for point in self.click_points:
      color = BLUE if point[2] == "left" else YELLOW
      adjusted_point = (point[0] + self.offset_x, point[1] + self.offset_y)
      pygame.draw.circle(self.screen, color, adjusted_point, 4);
      
    self.adjusted_origin_x = self.origin_x + self.offset_x
    self.adjusted_origin_y = self.origin_y + self.offset_y
    pygame.draw.circle(self.screen, RED, (self.adjusted_origin_x, self.adjusted_origin_y), 8)
    pygame.display.flip()
  
def main():
  visualizer = TrackVisualizer()
  running = True
  
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      elif event.type == pygame.MOUSEBUTTONDOWN:
        mouse_type = event.button
        x, y = event.pos
        visualizer.add_point(x, y, event.button)
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_s:
          visualizer.save_points()
        elif event.key == pygame.K_UP:
          visualizer.pan(0, -10)
        elif event.key == pygame.K_DOWN:
          visualizer.pan(0, 10)
        elif event.key == pygame.K_LEFT:
          visualizer.pan(-10, 0)
        elif event.key == pygame.K_RIGHT:
          visualizer.pan(10, 0)
          
    visualizer.draw()
  
  pygame.quit()  
  sys.exit()
  
if __name__ == "__main__":
  main()