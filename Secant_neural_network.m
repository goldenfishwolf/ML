function [ w1,w2,dw2,error1,error2 ] = Secant_neural_network( X,Y,output_node_num,hidden_node_num,hidden_layer_num,is_first_time,weight_1,weight_2, d_weight_1)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%each sample has a row vector input and a row label

if size(hidden_node_num) ~= hidden_layer_num 
	%if the size of hidden_node_num is not hidden_layer_num, then print error and return.
	disp('the size of hidden_node_num is not hidden_layer_num!')
return;
end

[szm,szn] = size(X);
W1 = cell(1,hidden_layer_num+1);
DW1 = cell(1,hidden_layer_num+1);
W2 = cell(1,hidden_layer_num+1);
DW2 = cell(1,hidden_layer_num+1);
z = cell(1,hidden_layer_num+2);
h_n = cat(2,szn,hidden_node_num);
h_n = cat(2,h_n,output_node_num);
sum = 0;

if is_first_time == 1
%if is first time call the function, then we should calculate w1
	for i = 1:hidden_layer_num+1
	%reshape the initial weight
		W1(1,i)={reshape(weight_1(sum:sum + h_n(i)*h_n(i+1)), h_n(i), h_n(i+1))};
		DW1(1,i)={zero(h_n(i),h_n(i+1))};
		%W(1,i)={rand(h_n(i),h_n(i+1))};
		%DW(1,i)={zero(h_n(i),h_n(i+1))};
		sum = sum + h_n(i)*h_n(i+1);
	end
	error1 = 0;

	for i=1:szm
	% calculate error and differential with each sample 
		x = (X(i,:))';
		y = (Y(i,:))';
		z(1) = {x};
		for j = 1:hidden_layer_num
			z(j+1) = {1./(1+exp(-diag((cell2mat(z(j))')*cell2mat(W1(1,j)))))};
		end
		z(hidden_layer_num+2) = {diag((cell2mat(z(hidden_layer_num+1))')*cell2mat(W1(1,hidden_layer_num+1)))};
		O = exp(cell2mat(z(hidden_layer_num+2)))/sum(exp(cell2mat(z(hidden_layer_num+2))));
		error1 = error1 - dot(y,(log(O)));
		tem = (sum(y)*O - y);
		DW1(1,hidden_layer_num+1) = {cell2mat(DW1(1,hidden_layer_num+1)) + (tem*(cell2mat(z(hidden_layer_num+2))'))};
		for j = hidden_layer_num+1:-1:1
			tem_z = cell2mat(z(j));
			tem = (cell2mat(W1(j))*tem).*(tem_z*(1.-tem_z));
			DW1(1,j) = {cell2mat(DW(1,j)) + tem*z(tem_z)'};
		end
	end
end
sum = 0;
for i = 1:hidden_layer_num+1
%reshape the initial weight
	W2(1,i)={reshape(weight_2(sum:sum + h_n(i)*h_n(i+1)), h_n(i), h_n(i+1) )};
	DW2(1,i)={zero(h_n(i),h_n(i+1))};
	sum = sum + h_n(i)*h_n(i+1);
end
error2 = 0;
for i=1:szm
% calculate error and differential with each sample 
	x = (X(i,:))';
	y = (Y(i,:))';
	z(1) = {x};
	for j = 1:hidden_layer_num
		z(j+1) = {1./(1+exp(-diag((cell2mat(z(j))')*cell2mat(W2(1,j)))))};
	end
	z(hidden_layer_num+2) = {diag((cell2mat(z(hidden_layer_num+1))')*cell2mat(W2(1,hidden_layer_num+1)))};
	O = exp(cell2mat(z(hidden_layer_num+2)))/sum(exp(cell2mat(z(hidden_layer_num+2))));
	error2 = error2 - dot(y,(log(O)));
	tem = (sum(y)*O - y);
	DW2(1,hidden_layer_num+1) = {cell2mat(DW2(1,hidden_layer_num+1)) + (tem*(cell2mat(z(hidden_layer_num+2))'))};
	for j = hidden_layer_num+1:-1:1
		tem_z = cell2mat(z(j));
		tem = (cell2mat(W2(j))*tem).*(tem_z*(1.-tem_z));
		DW2(1,j) = {cell2mat(DW2(1,j)) + tem*z(tem_z)'};
	end
end
w1=[];
dw1=[];
w2=[];
dw2=[];
if is_first_time == 1
	for i = 1:hidden_layer_num+1
		%reshape matrix into vector
		w1 = cat(2,w1,reshape(cell2mat(W1(1,i)),1,[]));
		dw1 = cat(2,dw1,reshape(cell2mat(DW1(1,i)),1,[]));
	end
else
	w1 = weight_1;
	dw1 = d_weight_1;
end
for i = 1:hidden_layer_num+1
%reshape matrix into vector
	w2 = cat(2,w2,reshape(cell2mat(W2(1,i)),1,[]));
	dw2 = cat(2,dw2,reshape(cell2mat(DW2(1,i)),1,[]));
end
temp = w2;
v=dot(dw2,(w1-w2))*(dw1-dw2);
v=v/norm(v);
w2 = w1-v;
w1 = temp;
end

