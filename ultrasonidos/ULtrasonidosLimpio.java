
; incluir el archivo donde están todos los parametros
.include "8515def.inc"        

;vectores de interrupciones
  		
  		rjmp Init
		reti
		reti			; External 1 interrupt  Vector 
		reti			; Timer 1 Capture  Vector 
		reti
		reti			; Timer 1 CompareB  Vector 
		reti			; Timer 1 Overflow  Vector 
		reti			; Timer 0 Overflow  Vector 
		reti			; SPI  Vector 
		reti			; UART Receive  Vector 
		reti			; UDR Empty  Vector 
		reti			; UART Transmit  Vector 
		reti			; Analogue Comparator  Vector 

.equ MASCARADE1 = 0b00001111
.equ   MASCARA_D  =0b11110000
.equ   MASCARA_A  =0b00001111
.equ   MASCARA1_En  =0b11110000
.equ   MASCARA2_EL_C  =0b11110000  
.equ   MASCARA3_EL_B  =0b00000000
.equ   MASCARA_ASCII = 0b00110000
.equ   MASCARA_PULLUP = 0b11110000

.equ SCLP =  0
.equ SDAP =  1
.equ b_dir = 0
.equ TWIrd = 1
.equ TWIwr = 0


;**** Global Register Variables ****

.def	TWIdelay= r16			; Delay loop variable
.def	TWIdata	= r17			; TWI data transfer register
.def	TWIadr	= r18			; TWI address and direction register
.def	TWIstat	= r19			; TWI bus status register
.def   temp =r26 
 
.DSEG

temporal: .BYTE 128


; se inicia el programa principal (segmento de código)                        
.CSEG   


; SUBRUTINA: sendbyte         
; espera a que se haya transmitido el byte 
        
sendbyte:
	push r16
sendbyte2:
	sbis usr,6   ;mira a ver si ya ha sido transmitido
		     ;el sexto bit de usr a 1
	rjmp sendbyte2
	in r16, usr
	andi r16,0b11011111 ;limpiar el sexto bit del registro
	out usr,r16
	pop r16
	ret        


;********************************
; SUBRUTINA ASCII
; parametros: r23 la parte alta del contador de medidas correspondiente
;	      r22 la parte baja
; salida
; r16 (centenas)
; r17 (decenas)
; r18 (unidades)
	
ASCII:	
	push r16
	push r17
	push r18
	push r19
	push r20
	push r21

	;empleo r16 (centenas), r17 (decenas) y r18 (unidades)

	cpi r23, 3
	brne r23_2
	cpi r22,132
	brlo menosde900
	ldi r16,9
	subi r22,132
	rjmp decenas
menosde900:
	cpi r22,32
	brlo menosde800
	ldi r16,8
	subi r22,32
	rjmp decenas
menosde800:		
	ldi r16,7
	subi r22,188
	sbci r23,2
	rjmp decenas
r23_2:
	cpi r23,2
	brne r23_1
	cpi r22,188
	brsh masde700
	cpi r22,88
	brsh masde600
masde500:
	subi r22,244
	sbci r23,1
	ldi r16,5
	rjmp decenas
	
masde600:	
	subi r22,88	
	ldi r16,6
	rjmp decenas
masde700:
	subi r22,188
	ldi r16,7
	rjmp decenas
r23_1:	
	cpi r23,1
	brne r23_0 
	cpi r22,244
	brsh masde500b
	cpi r22,144
	brsh masde400
	cpi r22,44
	brsh masde300
masde200:
	subi r22,200
	sbci r23,0
	ldi r16,2
	rjmp decenas

masde500b:
	ldi r16,5
	subi r22,244
	sbci r23,1
	rjmp decenas
	
masde400:	
	ldi r16,4
	subi r22,144
	sbci r23,1	
	rjmp decenas
masde300:
        ldi r16,3
        subi r22,44
	sbci r23,1
	rjmp decenas
r23_0:
	cpi r22,200
	brsh masde200b
	cpi r22,100
	brsh masde100
masde0:	
	ldi r16,0
	rjmp decenas
	
masde200b:
	subi r22,200
	ldi r16,2
	rjmp decenas
masde100:
	subi r22,100
	sbci r23,0
	ldi r16,1
	rjmp decenas
decenas:
	ldi r23,0	
	cpi r22,90
	brlo menosde90
	ldi r17,9
	subi r22,90
	rjmp unidades
menosde90:
        cpi r22,80
	brlo menosde80
	ldi r17,8
	subi r22,80
	rjmp unidades
menosde80:	
        cpi r22,70
	brlo menosde70
	ldi r17,7
	subi r22,70
	rjmp unidades
menosde70:	
        cpi r22,60
	brlo menosde60
	ldi r17,6
	subi r22,60
	rjmp unidades
menosde60:	
        cpi r22,50
	brlo menosde50
	ldi r17,5
	subi r22,50
	rjmp unidades
menosde50:	
        cpi r22,40
	brlo menosde40
	ldi r17,4
	subi r22,40
	rjmp unidades
menosde40:	
        cpi r22,30
	brlo menosde30
	ldi r17,3
	subi r22,30
	rjmp unidades
menosde30:	
        cpi r22,20
	brlo menosde20
	ldi r17,2
	subi r22,20
	rjmp unidades
menosde20:	
        cpi r22,10
	brlo menosde10
	ldi r17,1
	subi r22,10
	rjmp unidades
menosde10:	
	ldi r17,0
unidades:

	mov r18,r22
ldi r22,48
add r16,r22
out udr,r16
rcall sendbyte
add r17,r22
out udr,r17
rcall sendbyte
add r18,r22	
out udr,r18
rcall sendbyte
pop r21
 	pop r20	
 	pop r19
	pop r18
	pop r17
	pop r16
        ret


; SUBRUTINA: inicializaIO
 ; descripcion:
 ;              inicializa puertos de entrada o salida segun uso
 ;
 ;
 ; Nota: cuidado con el puerto C y las lineas como entradas y salidas
 ; hacerlo después de leer la identificacion
 
 inicializaIO:
         sbr r16, MASCARA_A
         out DDRA,r16  
         in r16, DDRD ; lee configuracion actual DDRA
         andi r16, ~MASCARA_D
         out DDRD,r16    
         in r16,DDRA ; lee configuracion actual DDRA
         cbr r16, MASCARA1_En
         out DDRA,r16   
         in r16,DDRC
         ldi r16,0
         out PORTC, r16
         sbr r16, MASCARA2_EL_C
         out DDRC,r16  
         in r16,DDRB
         sbr r16, MASCARA3_EL_B
         out DDRB,r16   ; 
         cbi DDRB,PB0 
         cbi DDRB,PB1  
        
  ret

enviaespacio:
 	ldi r24, 32	; Espacio
	out udr,r24
	rcall sendbyte
ret

;
; RUTINA QUE SIMPLEMENTE ESPERA UN TIEMPO PARA NORMALIZAR EL SISTEMA
;
espera:
;bucle para evitar cosas raras
push r16
push r17
push r18
push r19
push r20
push r21
push r22
push r23
ldi r20,1
ldi r21,4 
bucledel20:
ldi r19,1
bucledel19:
ldi r18,1
bucledel17:
ldi r17,255
bucledel16:
ldi r16,255
bucledel15:
dec r16
brne bucledel15
dec r17
brne bucledel16
dec r18
brne bucledel17
dec r19
brne bucledel19
inc r20
dec r21
brne bucledel20
pop r23        
pop r22
pop r21
pop r20
pop r19
pop r18
pop r17
pop r16 
ret

; FINAL ESPERA



;SUBRUTINA ESPERAPEQ 
; ESPERA PEQUE¥A PARA NO COLAPSAR LA IMPRESORA
;

esperapeq:
;bucle para evitar cosas raras
push r16
push r17
push r18
push r19
ldi r19,1
bucledel19x:
ldi r18,1
bucledel17x:
ldi r17,40
bucledel16x:
ldi r16,255
bucledel15x:
dec r16
brne bucledel15x
dec r17
brne bucledel16x
dec r18
brne bucledel17x
dec r19
brne bucledel19x
pop r19
pop r18
pop r17
pop r16 
ret





;
; COMIENZO DE EJECUCION DEL PROGRAMA
;
Init:
MAIN:
; se inicializa el programa
         ldi temp,low(RAMEND)
         out SPL,temp
         ldi temp,high(RAMEND)
         out SPH,temp ;init Stack Pointer
         rcall inicializaIO ; configuracion de los pines I/O   
; inicializar los registros del envio a la RS232	
	ldi r16,51
	out UBRR,r16       ;8 MHz, 9600 BAUD
; inicializar el UART en modo de transmision	
     	ldi r16,0x08		; UART Interrupt Enables and UART Settings 
	out UCR,r16		; -Settings: RX Disable, TX Enable, 9-Bit Disable 
        rcall inicializaIO ; configuracion de los pines I/O   
; cargar en los registros la unidad y cero
rcall espera
rcall espera
rcall espera

inicio:
rcall TWI_init

push r24
push r25

ldi r24,0xE0
ldi r25,16

buclePing:
; -- seleccionar el comando
; start
; escribir la dirección e0

mov TWIadr, r24
rcall TWI_start
; escribir el registro 0
ldi TWIdata, 0 
rcall TWI_do_transfer
; escribir el comando 81
ldi TWIdata, 82
rcall TWI_do_transfer
; stop
rcall TWI_stop

rcall espera

buclelectura:
; -- lee el byte
; poner el registro a 1
; start
; escribir la dirección
mov TWIadr, r24
rcall TWI_start
; escribir el registro
ldi TWIdata, 2
rcall TWI_write
; start
; la nueva dirección con un or 1
 inc TWIadr
 rcall TWI_rep_start
; escribir la dirección1
; leer el byte con i2cack = 0
sec
rcall TWI_do_transfer
rcall TWI_stop
; stop

; sacar el byte por impresora
push r23
push r22
ldi r23,0
ldi r22,255
rcall ASCII
ldi r22, 32
out udr,r22
rcall sendbyte
mov r22,r25
rcall ASCII
ldi r22, 32
out udr,r22
rcall sendbyte
mov r22, TWIdata
cpi r22,255
brne siguebien
dec r22
siguebien:

rcall ASCII
ldi r22, 32
out udr,r22
rcall sendbyte
pop r22
pop r23

; -- lee el byte
; poner el registro a 1
; start
; escribir la dirección
mov TWIadr, r24
rcall TWI_start
; escribir el registro
ldi TWIdata, 3
rcall TWI_write
; start
; la nueva dirección con un or 1
 inc TWIadr
 rcall TWI_rep_start
; escribir la dirección1
; leer el byte con i2cack = 0
sec
rcall TWI_do_transfer
rcall TWI_stop
; stop

; sacar el byte por impresora
push r23
push r22
mov r22, TWIdata
;esto por protocolo
ldi r23,0b11111110
and r22,r23
ldi r23,0
rcall ASCII
ldi r22, 10
out udr,r22
rcall sendbyte
ldi r22, 13
out udr,r22
rcall sendbyte
pop r22
pop r23
inc r24
inc r24
dec r25
brne buclePing2
pop r25
pop r24

rjmp inicio

buclePing2:
rjmp buclePing


; sacar el byte por impresora

; sacar el byte por impresora



;***************************************************************************
;*
;* FUNCTION
;*	TWI_hp_delay
;*	TWI_qp_delay
;*
;* DESCRIPTION
;*	hp - half TWI clock period delay (normal: 5.0us / fast: 1.3us)
;*	qp - quarter TWI clock period delay (normal: 2.5us / fast: 0.6us)
;*
;*	SEE DOCUMENTATION !!!
;*
;* USAGE
;*	no parameters
;*
;* RETURN
;*	none
;*
;***************************************************************************

TWI_hp_delay:
	ldi	TWIdelay,22
TWI_hp_delay_loop:
	dec	TWIdelay
	brne	TWI_hp_delay_loop
	ret

TWI_qp_delay:
	ldi	TWIdelay,11
TWI_qp_delay_loop:
	dec	TWIdelay
	brne	TWI_qp_delay_loop
	ret


;***************************************************************************
;*
;* FUNCTION
;*	TWI_rep_start
;*
;* DESCRIPTION
;*	Assert repeated start condition and sends slave address.
;*
;* USAGE
;*	TWIadr - Contains the slave address and transfer direction.
;*
;* RETURN
;*	Carry flag - Cleared if a slave responds to the address.
;*
;* NOTE
;*	IMPORTANT! : This funtion must be directly followed by TWI_start.
;*
;***************************************************************************

TWI_rep_start:
	sbi	DDRB,SCLP		; force SCL low
	cbi	DDRB,SDAP		; release SDA
	rcall	TWI_hp_delay		; half period delay
	cbi	DDRB,SCLP		; release SCL
	rcall	TWI_qp_delay		; quarter period delay


;***************************************************************************
;*
;* FUNCTION
;*	TWI_start
;*
;* DESCRIPTION
;*	Generates start condition and sends slave address.
;*
;* USAGE
;*	TWIadr - Contains the slave address and transfer direction.
;*
;* RETURN
;*	Carry flag - Cleared if a slave responds to the address.
;*
;* NOTE
;*	IMPORTANT! : This funtion must be directly followed by TWI_write.
;*
;***************************************************************************

TWI_start:				
	mov	TWIdata,TWIadr		; copy address to transmitt register
	sbi	DDRB,SDAP		; force SDA low
	rcall	TWI_qp_delay		; quarter period delay


;***************************************************************************
;*
;* FUNCTION
;*	TWI_write
;*
;* DESCRIPTION
;*	Writes data (one byte) to the TWI bus. Also used for sending
;*	the address.
;*
;* USAGE
;*	TWIdata - Contains data to be transmitted.
;*
;* RETURN
;*	Carry flag - Set if the slave respond transfer.
;*
;* NOTE
;*	IMPORTANT! : This funtion must be directly followed by TWI_get_ack.
;*
;***************************************************************************

TWI_write:
	sec				; set carry flag
	rol	TWIdata			; shift in carry and out bit one
	rjmp	TWI_write_first
TWI_write_bit:
	lsl	TWIdata			; if transmit register empty
TWI_write_first:
	breq	TWI_get_ack		;	goto get acknowledge
	sbi	DDRB,SCLP		; force SCL low

	brcc	TWI_write_low		; if bit high
	nop				;	(equalize number of cycles)
	cbi	DDRB,SDAP		;	release SDA
	rjmp	TWI_write_high
TWI_write_low:				; else
	sbi	DDRB,SDAP		;	force SDA low
	rjmp	TWI_write_high		;	(equalize number of cycles)
TWI_write_high:
	rcall	TWI_hp_delay		; half period delay
	cbi	DDRB,SCLP		; release SCL
	rcall	TWI_hp_delay		; half period delay

	rjmp	TWI_write_bit


;***************************************************************************
;*
;* FUNCTION
;*	TWI_get_ack
;*
;* DESCRIPTION
;*	Get slave acknowledge response.
;*
;* USAGE
;*	(used only by TWI_write in this version)
;*
;* RETURN
;*	Carry flag - Cleared if a slave responds to a request.
;*
;***************************************************************************

TWI_get_ack:
	sbi	DDRB,SCLP		; force SCL low
	cbi	DDRB,SDAP		; release SDA
	rcall	TWI_hp_delay		; half period delay
	cbi	DDRB,SCLP		; release SCL

TWI_get_ack_wait:
	sbis	PINB,SCLP		; wait SCL high 
					;(In case wait states are inserted)
	rjmp	TWI_get_ack_wait

	clc				; clear carry flag
	sbic	PINB,SDAP		; if SDA is high
	sec				;	set carry flag
	rcall	TWI_hp_delay		; half period delay
	ret


;***************************************************************************
;*
;* FUNCTION
;*	TWI_do_transfer
;*
;* DESCRIPTION
;*	Executes a transfer on bus. This is only a combination of TWI_read
;*	and TWI_write for convenience.
;*
;* USAGE
;*	TWIadr - Must have the same direction as when TWI_start was called.
;*	see TWI_read and TWI_write for more information.
;*
;* RETURN
;*	(depends on type of transfer, read or write)
;*
;* NOTE
;*	IMPORTANT! : This funtion must be directly followed by TWI_read.
;*
;***************************************************************************

TWI_do_transfer:
	sbrs	TWIadr,b_dir		; if dir = write
	rjmp	TWI_write		;	goto write data


;***************************************************************************
;*
;* FUNCTION
;*	TWI_read
;*
;* DESCRIPTION
;*	Reads data (one byte) from the TWI bus.
;*
;* USAGE
;*	Carry flag - 	If set no acknowledge is given to the slave
;*			indicating last read operation before a STOP.
;*			If cleared acknowledge is given to the slave
;*			indicating more data.
;*
;* RETURN
;*	TWIdata - Contains received data.
;*
;* NOTE
;*	IMPORTANT! : This funtion must be directly followed by TWI_put_ack.
;*
;***************************************************************************

TWI_read:
	rol	TWIstat			; store acknowledge
					; (used by TWI_put_ack)
	ldi	TWIdata,0x01		; data = 0x01
TWI_read_bit:				; do
	sbi	DDRB,SCLP		; 	force SCL low
	rcall	TWI_hp_delay		;	half period delay

	cbi	DDRB,SCLP		;	release SCL
	rcall	TWI_hp_delay		;	half period delay

	clc				;	clear carry flag
	sbic	PINB,SDAP		;	if SDA is high
	sec				;		set carry flag

	rol	TWIdata			; 	store data bit
	brcc	TWI_read_bit		; while receive register not full


;***************************************************************************
;*
;* FUNCTION
;*	TWI_put_ack
;*
;* DESCRIPTION
;*	Put acknowledge.
;*
;* USAGE
;*	(used only by TWI_read in this version)
;*
;* RETURN
;*	none
;*
;***************************************************************************

TWI_put_ack:
	sbi	DDRB,SCLP		; force SCL low

	ror	TWIstat			; get status bit
	brcc	TWI_put_ack_low		; if bit low goto assert low
	cbi	DDRB,SDAP		;	release SDA
	rjmp	TWI_put_ack_high
TWI_put_ack_low:			; else
	sbi	DDRB,SDAP		;	force SDA low
TWI_put_ack_high:
	rcall	TWI_hp_delay		; half period delay
	cbi	DDRB,SCLP		; release SCL
TWI_put_ack_wait:
	sbis	PINB,SCLP		; wait SCL high
	rjmp	TWI_put_ack_wait
	rcall	TWI_hp_delay		; half period delay
	ret


;***************************************************************************
;*
;* FUNCTION
;*	TWI_stop
;*
;* DESCRIPTION
;*	Assert stop condition.
;*
;* USAGE
;*	No parameters.
;*
;* RETURN
;*	None.
;*
;***************************************************************************

TWI_stop:
	sbi	DDRB,SCLP		; force SCL low
	sbi	DDRB,SDAP		; force SDA low
	rcall	TWI_hp_delay		; half period delay
	cbi	DDRB,SCLP		; release SCL
	rcall	TWI_qp_delay		; quarter period delay
	cbi	DDRB,SDAP		; release SDA
	rcall	TWI_hp_delay		; half period delay
	ret

;************************************** 





TWI_init:
	clr TWIstat
	out PORTB, TWIstat
	out DDRB, TWIstat 
ret

         

