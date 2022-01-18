% This function intends to generate average spectra based on a given amount
% of data. It will chunk the data into windows and get subwindows from each
% window. 

% Assumptions
% 1. Data is continuously at 10 Hz

% Input parameters:
% x: a vector representing a high-frequency quantity, typically 'w'
% Fs: a sampling frequency in Hertz
% window: a window of time over which spectra will be averaged, in minutes
% subwindow: the window will be chunked this many times

function [freq, spectra] = spectra_gabriel(x, Fs, window, subwindows)
    % Get number of minutes per window. Will be used for indexing data.
    window_size = window * 60 / (1/Fs);
    % Define number of windows
    windows = ceil(length(x)/window_size);
    % Chunk given data into windows (typically 30 mins)
    for index = 1:windows
        % Generate empty matrices for window data
        freq_matrix = [];
        spectra_matrix = [];
        % Define start and stop indices for vector slicing
        window_start = (index - 1) * window_size + 1;
        window_end = index * window_size;
        % Slice the vector corresponding to the window
        vec = x(window_start:window_end);
        % Chunk window data into subwindows
        for subindex = 1:subwindows
            % Range of indices in the window
            range = (window_end - window_start)/subwindows;
            % Subwindow indices
            subwindow_start = floor((subindex - 1) * range + 1);
            subwindow_end = floor(subindex * range);
            % Slice the vector to get a subwindow vector
            subvec = vec(subwindow_start:subwindow_end);
            % Run the power spectral script
            [Pxx, f] = powerspec_prathap(subvec, Fs);
            % Append vectors to each matrix
            freq_matrix = [freq_matrix; f];
            spectra_matrix = [spectra_matrix; Pxx];
        end
        % Get window average of the spectra
        freq = mean(freq_matrix);
        spectra = mean(spectra_matrix);
    end