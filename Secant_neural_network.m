function [ DW1,DW2,DW3,error ] = Secant_neural_network( X,Y,output_node_num,hidden_node_num,hidden_layer_num )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%each sample has a row vector input and a row label

if size(hidden_node_num) ~= hidden_layer_num 
	%if the size of hidden_node_num is not hidden_layer_num, then print error and return.
	disp('the size of hidden_node_num is not hidden_layer_num!')
return;
end

[szm,szn] = size(X);
W = cell(1,hidden_layer_num+1);
DW = cell(1,hidden_layer_num+1);
z = cell(1,hidden_layer_num+2);
h_n = cat(2,[szn],hidden_node_num);
h_n = cat(2,hidden_node_num,[output_node_num])
for i = 1:hidden_layer_num+1
%initial weight
	W(1,i)={rand(h_n(i),h_n(i+1))};
	DW(1,i)={zero(h_n(i),h_n(i+1))};
end
error = 0;

for i=1:szm
	x = (X(i,:))';
	y = (Y(i,:))';
	z(1) = {x};
	for j = 1:hidden_layer_num
		z(j+1) = {1./(1+exp(-diag((z(j)')*W(1,j))))};
	end
	z(hidden_layer_num+2) = {diag((z(hidden_layer_num+1)')*W(1,hidden_layer_num+1))};
	O = exp(z(hidden_layer_num+2))/sum(exp(z(hidden_layer_num+2)));
	error = error - dot(y,(log(O)));
	tem = (sum(y)*O - y);
	DW(1,hidden_layer_num+1) = {cell2mat(DW(1,hidden_layer_num+1)) + (tem*(z(hidden_layer_num+2)'))};
	for j = hidden_layer_num+1:-1:1
		tem_z = cell2mat(z(j));
		tem = (cell2mat(W(j))*tem).*(tem_z*(1.-tem_z));
		DW(1,j) = {cell2mat(DW(1,j)) + tem*z(tem_z)'};
	end
	
end
reshape()
end

