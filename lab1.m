% Computer Vision
% LAB 1 
% Cars Detection

imageFiles = dir('./Images/Images/*.jpg');
numImages = length(imageFiles);

%disp(numImages);

collection = cell(1, numImages);
for i = 1:numImages
    collection{i} = imread(fullfile('./Images/Images/', imageFiles(i).name));
end

% Convert to grayscale
collection = cellfun(@(img) rgb2gray(img), collection, 'UniformOutput', false);

% Split into training and test sets
Train = collection(1:150);
Test = collection(151:end);

% Compute mean and standard deviation for training set
mean_image = mean(cat(3, Train{:}), 3);
std_image = std(double(cat(3, Train{:})), 0, 3);

% Display mean and standard deviation images
figure;
imshow(mean_image, []);
title('Mean Image');

figure;
imshow(std_image, []);
title('Standard Deviation Image');

% Image Segmentation (Background Removal)
%Segmentation = cellfun(@(img) abs(double(img) - mean_image), Test, 'UniformOutput', false);
Test = cellfun(@double, Test, 'UniformOutput', false);
imshow(Test{1},[])

Segmentation = cellfun(@(img) abs(double(img) - double(mean_image)), Test, 'UniformOutput', false);
NormalizedSegmentation = cellfun(@(img) img / max(img(:)), Segmentation, 'UniformOutput', false);
Binary = cellfun(@(img) img > 0.20, NormalizedSegmentation, 'UniformOutput', false);
% Calculate combined term for all images
combined_term = beta + alpha * std_image;
min_val = min(combined_term);  
max_val = max(combined_term);
normalized_combined_term = (combined_term - min_val) / (max_val - min_val);
Binary_2 = cellfun(@(img) img > normalized_combined_term, NormalizedSegmentation, 'UniformOutput', false);

% Choose an index for the image you want to display
displayedImageIndex = 1; 
% Display the original and segmented images side by side
figure;

% Original Image
subplot(2, 2, 1);
imshow(Test{displayedImageIndex},[]);
title('Original Image');

% Segmented Image
subplot(2, 2, 2);
imshow(Segmentation{displayedImageIndex},[]);
title('Segmented Image');

% Binary Segmented Image
subplot(2, 2, 3);
imshow(Binary{displayedImageIndex},[]);
title('Binary Segmented Image');

% Binary_2 Model Improved Segmented Image
subplot(2, 2, 4);
imshow(Binary_2{displayedImageIndex},[]);
title('Binary_2 Model Improved Segmented Image');




