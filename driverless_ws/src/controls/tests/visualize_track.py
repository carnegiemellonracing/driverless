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
    self.active_chunk = None
    self.chunks = []
        
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
    filename = f"data/track_points_{timestamp}.txt"
    
    with open(filename, 'w') as file:
      if self.chunks:
        for idx, chunk in enumerate(self.chunks, 1):
          file.write(f"Chunk {idx}:\n")
          for point in chunk["points"]:
            x_meters = point[0] / self.scale
            y_meters = point[1] / self.scale
            file.write(f"{x_meters},{y_meters},{point[2]}\n")
          file.write("\n")
      else:
        file.write("All Points:\n")
        for point in self.click_points:
          x_meters = point[0] / self.scale
          y_meters = point[1] / self.scale
          file.write(f"{x_meters},{y_meters},{point[2]}\n")
          
    print(f"Points saved to {filename}\n")
      
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
      
    for chunk in self.chunks:
      start_x, start_y = chunk["start"]
      end_x, end_y = chunk["end"]
      top_left = (min(start_x, end_x) + self.offset_x, min(start_y, end_y) + self.offset_y)
      width = abs(end_x - start_x)
      height = abs(end_y - start_y)
      pygame.draw.rect(self.screen, WHITE, (*top_left, width, height), 2)
      
    if self.active_chunk:
      start_x, start_y = self.active_chunk["start"]
      mouse_x, mouse_y = pygame.mouse.get_pos()
      end_x, end_y = mouse_x - self.offset_x, mouse_y - self.offset_y
      top_left = (min(start_x, end_x) + self.offset_x, min(start_y, end_y) + self.offset_y)
      width = abs(end_x - start_x)
      height = abs(end_y - start_y)
      pygame.draw.rect(self.screen, WHITE, (*top_left, width, height), 2)

    self.adjusted_origin_x = self.origin_x + self.offset_x
    self.adjusted_origin_y = self.origin_y + self.offset_y
    pygame.draw.circle(self.screen, RED, (self.adjusted_origin_x, self.adjusted_origin_y), 8)
      
    pygame.display.flip()
      
      
  def start_chunk(self):
    mouse_x, mouse_y = pygame.mouse.get_pos()
    self.active_chunk = {"start": (mouse_x - self.offset_x, mouse_y - self.offset_y), "end": None, "points": []}
    
  def finalize_chunk(self):
    mouse_x, mouse_y = pygame.mouse.get_pos()
    self.active_chunk["end"] = (mouse_x - self.offset_x, mouse_y - self.offset_y)
  
    x1, y1 = self.active_chunk["start"]
    x2, y2 = self.active_chunk["end"]
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    
    self.active_chunk["points"] = [
      point for point in self.click_points if x_min <= point[0] <= x_max and y_min <= point[1] <= y_max
    ]
    self.chunks.append(self.active_chunk)
    self.active_chunk = None
    
  def clear_chunks(self):
    self.chunks = []
    
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
        elif event.key == pygame.K_d:
          if not visualizer.active_chunk:
            visualizer.start_chunk()
          else:
            visualizer.finalize_chunk()
        elif event.key == pygame.K_c:
          visualizer.clear_chunks()
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