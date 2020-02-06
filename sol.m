img_g = imread("example_2.jpeg");
img_d = double(img_g);

[N,M] = size(img_g);

x = zeros(N,M);
for i = 1:1:N
    x(i,:) = ones(1,M)*i-ceil(N/2);
end
y = zeros(N,M);
for i = 1:1:M
    y(:,i) = ones(N,1)*i-ceil(M/2);
end
r = floor(min(N,M)/2.2);
circle = x.^2 + y.^2 < r.^2;
circle = abs(imgradient(circle,'intermediate'))~=0;

shape = circle;

iterations = 300;
dt = 1;
new_shape = sinld(img_d,dt,iterations,shape);

f = figure;
subplot(1,2,1)

s = img_d/256;
img_color = cat(3,s,s,s);
imshow(img_color.*~shape)
hold all
[M,c]= contour(shape,'red','LineWidth',0.1);
title('Original Segmentation')

subplot(1,2,2)
img_color = cat(3,s,s,s);
imshow(img_color.* ~new_shape)
hold all
[M,c]= contour(new_shape,'red','LineWidth',0.1);
title('Segmentation after Geodesic Active Contours')

saveas(f,'sol.png');

function [Iss,Isn,Isw,Ise]=shifts(I)
    Iss = circshift(I,1,1); %   matrix shifted to the south
    Isn = circshift(I,-1,1);%   matrix shifted to the north
    Isw = circshift(I,-1,2);%   matrix shifted to the west
    Ise = circshift(I,1,2);%    matrix shifted to the east
end

function [upper,diagonal,lower] = diagonals(C,dt)

    Csw = circshift(C,-1,2);
    Cse = circshift(C,1,2);

    upper = - dt * 2 * (C + Csw);
    lower = - dt * 2 * (Cse + C);
    diagonal = 2 + dt * ( 4 * C + 2 * Csw + 2 * Cse);
end

function shape=sinld(I,dt,iterations,shape_)
    
    shape = shape_;

    [N,M] = size(I);
    
    [Iss,Isn,Isw,Ise]=shifts(I);
        
    In = (Iss-Isn)/2;
    Ie = (Isw-Ise)/2;
        
    C = 1 ./ (1 + (In.^2 + Ie.^2));         
        
    % flatten C row-wise
        
    Crow = reshape(C.',1,[]);
        
    [upper_row,diagonal_row,lower_row] = diagonals(Crow,dt);
    
    % flatten C column-wise
        
    Ccol = reshape(C,1,[]);
        
    [upper_col,diagonal_col,lower_col] = diagonals(Ccol,dt);                
    
    for it = 1:1:iterations 
        
        % shape to distance
        
        inner_mask = imfill(shape,[floor(N/2) floor(M/2)])-shape;

        distance = bwdist(shape);
        distance = distance .* ~inner_mask + distance .* inner_mask .* (-1);
        
        phi = distance;
        

        tmp = 0;
        
        % flatten phi row-wise

        sol = tridiag(diagonal_row,lower_row,upper_row,reshape(phi.',1,[]));
        
        tmp = tmp + reshape(sol,M,N)';
        
        % flatten phi column-wise
        
        sol = tridiag(diagonal_col,lower_col,upper_col,reshape(phi,1,[]));
        
        tmp = tmp + reshape(sol,N,M);
        
        phi = tmp; 
        
        % distance to shape
        
        shape = abs(phi)<1;
        
    end
    
    shape_ = imfill(shape,[1 1]);
    shape = abs(imgradient(shape_,'intermediate'))~=0;
end



